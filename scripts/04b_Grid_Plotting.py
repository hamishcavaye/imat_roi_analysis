"""
A script that takes previously processed .csv data from a neutron 
radiography run and allows the plotting of a heatmap of transmission changes
based on a grid.
 
Copyright (C) 2026  Hamish Cavaye

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.

To-do:
    - Provide a flag in class_defs / cfg that allows you to plot in
      Increased transmission, rather than Decreased transmission
"""

import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.patches as patches
import matplotlib.cm as cm
import pandas as pd
import numpy as np
import time, json
from scipy.signal import savgol_filter
from dataclasses import asdict
from pathlib import Path
from matplotlib_scalebar.scalebar import ScaleBar
from skimage import io
from concurrent.futures import ProcessPoolExecutor, as_completed
from utils import create_outputdir, save_runtime

# Import the variables from the experiment config file
# Make sure to add the folder with this file to the sys.path and that no other
# paths to config files are present!
from cfg import experiment, run_data, grid_figs

# Choose which grid timelapse to plot (counts from 0!)
figpars = grid_figs[0]

###############################################################################

# Get the start time so we can report total processing time later
print("---")
print("Starting data analysis...")
print(" ")
total_start_time = time.time()

def crop_image_and_grid(image, crop_limits, grid):
    """
    image: numpy array (H, W) or (H, W, 3)
    crop_limits: ((xmin, xmax), (ymax, ymin))  <-- your format
    grid: array of (x, y, w, h) tuples

    Returns:
        cropped_image, updated_grid
    """
    # Unpack the reversed y-limits
    (x0, x1), (ymax, ymin) = crop_limits

    # Convert into correct numpy slicing order
    y0 = ymin
    y1 = ymax

    # Crop the actual image (numpy requires y0 < y1)
    cropped = image[y0:y1, x0:x1]

    # Shift grid so new origin is (0, 0)
    new_grid = np.empty_like(grid)
    for r in range(grid.shape[0]):
        for c in range(grid.shape[1]):
            gx, gy, w, h = grid[r, c]

            # gy is measured from original image top-left,
            # so subtract ymin (the new top)
            new_x = gx - x0
            new_y = gy - y0

            new_grid[r, c] = (new_x, new_y, w, h)

    return cropped, new_grid

# Globals that each process will set once in _init_worker
_G = {}

def _init_worker(ref_image, grid_array, 
                 cmap_name, use_log, centre, vmin, vmax, figpars_min, output_dir):
    """
    Initializer runs once per worker process. Populate a small global dict.
    """
    # Rebuild colormap & normalisation in the worker
    try:
        from matplotlib import colormaps
        cmap = colormaps[cmap_name]
    except Exception:
        import matplotlib.pyplot as plt
        cmap = plt.get_cmap(cmap_name)

    if use_log:
        norm = colors.LogNorm(vmin=vmin, vmax=vmax)
    elif col_norm["norm"] == "twinslope":
        norm = colors.TwoSlopeNorm(vcenter=centre, vmin=vmin, vmax=vmax)
    else:
        norm = colors.Normalize(vmin=vmin, vmax=vmax)

    # Keep only the minimal fig parameters we need
    _G["ref_img"] = ref_image
    _G["grid"] = grid_array
    _G["cmap"] = cmap
    _G["norm"] = norm
    _G["p"] = figpars_min
    _G["outdir"] = str(output_dir)

def _process_one_frame(args):
    """
    Render a single frame and save it to disk.
    args: (frame_idx, elapsed_value, row_vector_of_values, colormap_side_label)
    """
    idx, elapsed_value, intensity_row = args

    ref_img = _G["ref_img"]
    grid = _G["grid"]
    cmap = _G["cmap"]
    norm = _G["norm"]
    p = _G["p"]
    outdir = _G["outdir"]

    # Create figure
    fig, ax = plt.subplots(figsize=p["figure_size"])
    ax.imshow(ref_img, cmap="gray",
              vmin=np.min(ref_img),
              vmax=np.percentile(ref_img, 99))  # 99th pct clipping for bright spots

    # Paint patches
    k = 0
    for r in range(grid.shape[0]):
        for c in range(grid.shape[1]):
            gx, gy, w, h = grid[r, c]
            intensity = intensity_row[k]  # value already transformed in parent
            color = cmap(norm(intensity))
            rect = patches.Rectangle((gx, gy), w, h,
                                     linewidth=0,
                                     facecolor=color,
                                     alpha=p["heatmap_alpha"])
            ax.add_patch(rect)
            k += 1

    # Timestamp
    unit = "min" if p["conv_mins"] else "s"
    ax.annotate(f"{elapsed_value:.0f} {unit}",
                xy=p["timestamp_loc"],
                xycoords="axes fraction",
                fontsize=p["timestamp_font_size"],
                color=p["timestamp_colour"])

    # Colorbar
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    cbar = plt.colorbar(sm, ax=ax, shrink=p["bar_shrink_factor"])
    cbar.set_label(p["cbar_axis_title"], size=p["axes_label_font_size"])
    cbar.ax.tick_params(labelsize=p["tick_font_size"])

    # Scale bar (optional)
    if p["scale_bar"]:
        sb = ScaleBar(p["scale_bar"][0],
                      "mm",
                      fixed_units="mm",
                      fixed_value=p["scale_bar"][1],
                      location=p["scale_bar_pos"][0],
                      color=p["scale_bar_pos"][1],
                      box_alpha=0,
                      font_properties={"size": 30},
                      border_pad=1)
        ax.add_artist(sb)

    plt.axis("off")

    # Save filename
    if p["conv_mins"]:
        filename = str(int(elapsed_value * 60)).zfill(5) + ".png"
    else:
        filename = str(int(elapsed_value)).zfill(5) + ".png"
    filename = p["fig_name"] + " " + filename

    fig.canvas.draw()
    fig.savefig(Path(outdir) / filename, bbox_inches="tight")
    plt.close(fig)
    return idx, filename

# Define a function that converts to concentration from transmission
# percentage if Beer-Lambert is being used
# Takes a fractional grey value (e.g. 0.99 for 1% loss of transmission)
def beer_lambert_conv(delta, thickness, total_crosssection):
    conc = 1000 * (-np.log(delta)/(1 * thickness)/total_crosssection/1e-24/6.02e23)
    return conc

# Get the run .csv files from the config file
input_dir = Path(experiment.exp_outputs_path)
run = run_data[figpars.run]
run_csv = input_dir / run.output_name / f"{run.output_name}_greyvalues.csv"

# If the output path doesn't exist, create it
output_dir = Path(experiment.exp_outputs_path) / figpars.output_subdir
create_outputdir(output_dir)

# Save the parameters and script as .txt for later reference
print("---")
print("Saving parameters and script to files for reference...")
print(" ")
snippet = Path(__file__).read_text()
(output_dir / "analysis_script.txt").write_text(snippet)
exp_dict = asdict(experiment)
exp_dict["reference_frame_range"] = list(experiment.reference_frame_range)
(output_dir / "experiment_parameters.txt").write_text(json.dumps(exp_dict, indent=4))
fig_dict = asdict(figpars)
fig_dict['cmap'] = figpars.cmap.name
(output_dir / "figure_parameters.txt").write_text(json.dumps(fig_dict, indent=4))

# Start loading the data
print("---")
print("Loading the previously saved data...")
print(" ")
# Create some empty dicts to keep the dataframes from each run in
runs_df = {}
# Create an empty dataframe to hold the concatenated data
total_data = pd.DataFrame()

# Read the run into a dataframe from the .csv, converting the time from
# a string to a timestamp data type
df = pd.read_csv(run_csv)
df['Absolute Time'] = pd.to_datetime(df['Absolute Time'])
total_data = df

# We need to fix the Elapsed time column for the concatenated data to be
# relative to the start of the first run and then sort by Absolute Time
total_data = total_data.drop(['Elapsed Time'], axis=1)
elapsed_time_list = []
for a_time in total_data['Absolute Time']:
    e_time = (a_time - total_data['Absolute Time'].iloc[0]).total_seconds()
    if figpars.conv_mins:
        e_time = e_time/60
    elapsed_time_list.append(e_time)
total_data.insert(loc=1, column='Elapsed Time', value = pd.Series(elapsed_time_list))
total_data.sort_values('Absolute Time', inplace=True)
# total_data = total_data.copy()      # Generates a fresh, unfragmented dataframe

# If require the smoothed data, smooth it here
if figpars.smooth:
    window = figpars.savgol_window
    if window % 2 == 0:
        window += 1
        print(f"Adjusted Savitzky-Golay window to {window} (must be odd).")
    if figpars.savgol_polyorder < window:
        print(f"Smoothing values using a Savitzky-Golay signal smoothing algorithm. "
              f"Using a window of {window} and polynomial order of "
              f"{figpars.savgol_polyorder}.")
        print("")
        for col_name in total_data.columns:
            if col_name != "Absolute Time" and col_name != 'Elapsed Time':
                temp_series = savgol_filter(total_data[col_name],
                                            window,
                                            figpars.savgol_polyorder)
                total_data[col_name] = temp_series
                del temp_series
    else:
        raise ValueError(f"Error: Savitzky-Golar polynomial order is less than the window and should be greater: Window = {window}, Order = {figpars.savgol_polyorder}")

# Create a dataframe for data normalised to the first frame for each ROI
total_data_norm = total_data.copy()
for column in total_data.columns:
    if column != "Absolute Time" and column != 'Elapsed Time':
        total_data_norm[column] = total_data[column]/total_data[column][0]

# If plotting a cropped time regime (most common), crop the data here
if figpars.crop_time:
    print("---")
    print("")
    print("Cropping the data to the chosen time period: " + str(figpars.start_time) + " to " + str(figpars.end_time))
    print(" ")
    # First convert the start/end strings to Timestamps
    start_time = pd.Timestamp(figpars.start_time)
    end_time = pd.Timestamp(figpars.end_time)
    
    #####
    # Convert this using the n-data mask code??
    #####
    
    # Define a small function that returns only the data between start/end times
    def crop_the_data(data, start, end):
        data = data[data['Absolute Time'] > start]
        data = data[data['Absolute Time'] < end]
        return data
    
    # Actually perform the data crop for both datasets
    total_data = crop_the_data(total_data, start_time, end_time)
    total_data_norm = crop_the_data(total_data_norm, start_time, end_time)
    
    # If you want the first point in the cropped dataset to have elapsed time = 0
    # then you need to change that here
    if figpars.tzero:
        total_data['Elapsed Time'] = total_data['Elapsed Time'] - total_data['Elapsed Time'].iloc[0]
        total_data_norm['Elapsed Time'] = total_data_norm['Elapsed Time'] - total_data_norm['Elapsed Time'].iloc[0]
    
    # If you want normalisation to the start of the cropped time rather than the 
    # beginning of the whole experiment then you need to perform the normalisation again
    if figpars.normalise_tzero:
        total_data_norm = total_data.copy()
        for column in total_data.columns:
            if column != "Absolute Time" and column != 'Elapsed Time':
                total_data_norm[column] = total_data[column]/total_data[column].iloc[0]

# Get the grid parameters
grid_position = experiment.grid_position
# Calculate the width and height of each grid element
width = grid_position[2] / experiment.grid_cols
height = grid_position[3] / experiment.grid_rows
# Initiate an array of the correct size and shape
grid = np.empty((experiment.grid_rows, experiment.grid_cols), dtype=object)
for r in range(experiment.grid_rows):
    for c in range(experiment.grid_cols):
        top_left_x = grid_position[0] + c * width
        top_left_y = grid_position[1] + r * height
        grid[r, c] = (top_left_x, top_left_y, width, height)
grid_elements = []
for r, row in enumerate(grid):
    for c, col in enumerate(row):
        grid_elements.append(str(r) + "," + str(c))

# Normalise the data if desired
if figpars.normalise:
    data_to_plot = total_data_norm.copy()
else:
    data_to_plot = total_data.copy()        

# Load the reference image for plotting over
ref_image = io.imread(figpars.ref_image)

if figpars.crop_image:
    print("Cropping image")
    print(" ")
    ref_image, grid = crop_image_and_grid(
        ref_image,
        figpars.crop_limits,
        grid
    )

# Build an array of the values to plot for all frames, ordered by grid_elements
# This avoids indexing per patch inside workers.
values_matrix = data_to_plot[grid_elements].to_numpy()  # shape: (n_frames, n_patches)

# Apply the per-pixel transform:
if figpars.bl:
    values_matrix = beer_lambert_conv(values_matrix,
                                      figpars.samp_thickness,
                                      figpars.total_xsection
                                      )
elif figpars.invert_scale:
    # This means the result being plotted is "Reduced Transmission %"
    values_matrix = (1.0 - values_matrix) * 100.0

# Prepare minimal parameters (avoid passing the whole dataclass into workers)
col_norm = figpars.cmap_norm
cmap_name = figpars.cmap.name                         # safer to pass name
use_log = True if col_norm["norm"] == "log" else False
vmin, vmax = col_norm["vmin"], col_norm["vmax"]
if col_norm["norm"] == "twinslope":
    centre = col_norm["centre"]
else: centre = None
figpars_min = {
    "figure_size": figpars.figure_size,
    "heatmap_alpha": figpars.heatmap_alpha,
    "conv_mins": figpars.conv_mins,
    "timestamp_loc": figpars.timestamp_loc,
    "timestamp_font_size": figpars.timestamp_font_size,
    "timestamp_colour": figpars.timestamp_colour,
    "bar_shrink_factor": figpars.bar_shrink_factor,
    "axes_label_font_size": figpars.axes_label_font_size,
    "tick_font_size": figpars.tick_font_size,
    "scale_bar": figpars.scale_bar,
    "scale_bar_pos": figpars.scale_bar_pos,
    "fig_name": figpars.fig_name,
    "cbar_axis_title": figpars.bl_axis_title if figpars.bl else figpars.cbar_title,
}

# Create the per-frame argument tuples
elapsed = data_to_plot["Elapsed Time"].to_numpy()  # vector of elapsed times
tasks = [(i, float(elapsed[i]), values_matrix[i, :]) for i in range(values_matrix.shape[0])]

print("---")
print("Starting data processing...")
print(" ")

# Start the worker pool
max_workers = None  # or set to an int; None = os.cpu_count()
with ProcessPoolExecutor(
    max_workers=max_workers,
    initializer=_init_worker,
    initargs=(
        ref_image,     # already cropped
        grid,          # already shifted
        cmap_name, use_log, centre, vmin, vmax,
        figpars_min, output_dir,
    )
) as ex:
    futures = [ex.submit(_process_one_frame, t) for t in tasks]
    total = len(futures)
    completed = 0
    next_update = 10  # show updates at 10%, 20%, ..., 100%
    for fut in as_completed(futures):
        idx, fname = fut.result()
        completed += 1
        pct = (completed / total) * 100
        if pct >= next_update:
            print(f"Progress: {next_update}% ({completed}/{total})")
            next_update += 10


# Finally save a file to say how long the total script ran for
total_run_time = time.time() - total_start_time
print("--- Total analysis finished in %s seconds ---" % (total_run_time))
save_runtime(output_dir, "analysis_script_timer.txt", total_run_time)

