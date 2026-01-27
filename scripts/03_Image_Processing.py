"""
A script to take uncalibrated data frames and apply outlier removal as well as
dark subtraction and flat-fielding.

The resulting images can be saved whole, and the average grey value for various
ROIs are exported to .csv for further processing. Alternatively to the
various ROIs, one can generate a grid of regions in a given area to plot a 
heatmap later.

Regions should also be designated for direct beam normalisation and regions 
for an internal standard/calibration material can be given.

To-do list:
    - Pre-calculated slices for grids like ROI list?
    - Make text and annotations relative size rather than absolute?
    
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
"""

import time, csv, json
import numpy as np
from skimage import io
from scipy import ndimage
import dateutil.parser as dparser
from natsort import natsorted 
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from concurrent.futures import ProcessPoolExecutor, as_completed
from matplotlib_scalebar.scalebar import ScaleBar
from dataclasses import asdict
from pathlib import Path

# Import the variables from the experiment config file
# Make sure to add the folder with this file to the sys.path and that no other
# paths to config files are present!
from cfg import experiment, run_data

# Set which run to analyse
run = 5
run = run_data[run]

###############################################################################

output_dir = Path(experiment.exp_outputs_path) / run.output_name

# If the output path doesn't exist, create it
if not output_dir.exists():
    print("---")
    print("Output directory doesn't exist, creating it.")
    print(" ")
    output_dir.mkdir(parents=True, exist_ok=True)
    
# Save the parameters and script as .txt for later reference
print("---")
print("Saving parameters and script to files for reference...")
print(" ")
snippet = Path(__file__).read_text()
(output_dir / "image_processing_script.txt").write_text(snippet)
exp_dict = asdict(experiment)
exp_dict["reference_frame_range"] = list(experiment.reference_frame_range)
(output_dir / "experiment_processing_parameters.txt").write_text(json.dumps(exp_dict, indent=4))
run_dict = [asdict(a) for a in run_data[1:]]
(output_dir / "run_parameters.txt").write_text(json.dumps(run_dict, indent=4))

# Timestamp the start of the process
total_start_time = time.time()

# A helper function to convert the 4-tuple for an ROI into a slice
def roi_to_slice(x, y, w, h):
    return (slice(int(y), int(y+h)), slice(int(x), int(x+w))) 

if experiment.roi_grid:
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
    roi_groups = grid
else:
    roi_groups = experiment.roi_groups

# Import the coordinates for a region with fixed brightness for calibration
calib_topleft = experiment.calib_topleft

# Import the coordinates for (a) region(s) containing an internal standard
standard_rois = experiment.standard_rois

# Import a list of colours to use for the ROIs
colours = experiment.colours

# Load the master dark and master flat, then generate an inverse flat with
# epsion of 1e-6 to avoid any "divide by zero" issues
dark_flat_path = Path(experiment.exp_outputs_path) / experiment.dark_flat_output_path
master_dark = io.imread(dark_flat_path / "master_dark.tif")
master_flat_dark_corrected = io.imread(dark_flat_path / "dark_subtracted_master_flat.tif")
inverse_master_flat = 1.0 / np.maximum(master_flat_dark_corrected.astype(np.float32, copy=False), 1e-6)

def outlier_removal(image, bright_threshold, dark_threshold, radius):
    """
    Process a given image to remove outliers based on a bright pixel
    and dark pixel threshold values and a radius.
    """
    median_image = ndimage.median_filter(image, size=(radius,radius))
    outliers_bright = (image - median_image) > bright_threshold
    outliers_dark = (median_image - image) > dark_threshold
    result = np.where(outliers_bright | outliers_dark, median_image, image)
    return result

# Get the full list of filenames and truncate if necessary
print("---")
print("Obtaining filenames for processing...")
print(" ")
datapath = Path(experiment.raw_data_path) / run.data_subdir
#for file in datapath:
# filenames = glob.glob(datapath + '*.tif')
filenames = Path(datapath).glob('*.tif')
filenames = natsorted(filenames)
if run.truncate_image_list:
    filenames = filenames[run.truncate_image_list[0] : run.truncate_image_list[1]]
if run.frame_skip:
    filenames = filenames[0::run.frame_skip]
print("{} files have been found for processing.".format(str(len(filenames))))
print(" ")

# Extract the frame times from the CSV logs
print("---")
print("Extracting timestamps from log file...")
print(" ")
with open(Path(experiment.csv_path) / run.csvlog, newline='') as csvfile:
    reader = csv.DictReader(csvfile,
                            delimiter=',',
                            fieldnames = ['time', 'type', 'count', 'monbefore', 'monafter']
                            )
    times = []  # A list of the absolute times for each frame
    for row in reader:
        if "Stopping" not in row['type']:
            times.append(dparser.parse(row['time']))
    if run.frame_skip:
        times = times[0::run.frame_skip]
    if run.truncate_log:
        times = times[run.truncate_log[0] : run.truncate_log[1]]

# Generate the elapsed times since the run start for each frame
starttime = times[0]
timestamp = [(t-starttime).total_seconds() for t in times]

# Initialise an empty list to put all the grey values in
all_greyvalues = []
# Sequentially go through each file and process it
print("---")
print("Calibrating and processing all data frames...")
if experiment.save_full_frames:
    print("Including saving full frames to the output directory.")
if experiment.save_reference_frame:
    print("Including a reference frame with and without ROI markers")
    reference_frame = []
print(" ")

# This is the function that is called in the parallel processing for each frame
# It is passed the frame number (from enumerate(filenames)) and the frame itself
# It uses a number of "global" variables
def process_frame_and_save(i, frame):
    im = io.imread(frame)

    # Outlier removal via the function, above
    if experiment.remove_outliers:
        im = outlier_removal(im, experiment.bright_threshold, experiment.dark_threshold, experiment.outlier_radius)

    # Dark and flat frame calibration
    im = (im - master_dark) * inverse_master_flat

    # Get the average grey value in the calibration/normalisation region and then 
    # normalise the image so that this equals 1
    cal_roi = np.mean(im[calib_slice])
    im = im / cal_roi

    # Determine if you are processing a simple ROI list or a full grid
    if experiment.roi_grid:
        # Initialise a list to put the grey values for each grid element in
        # then iterate over all grid elements to get the average grey value in each
        average_greyvalues = []
        for row in roi_groups:
            for col in row:
                roi_bottomright = (col[0]+col[2], col[1]+col[3])
                roi = im[int(col[1]):int(roi_bottomright[1]), int(col[0]):int(roi_bottomright[0])]
                average_greyvalues.append(np.mean(roi))
    else:
        # Initialise a list to put the grey values for each ROI in
        # then iterate over all the ROIs to get the average grey value in each
        average_greyvalues = []
        for s in roi_slices:
            average_greyvalues.append(np.mean(im[s]))
    
    # If you have an internal standard ROI to process, do that here and append 
    # the results
    if len(standard_rois) > 0:
        for s in std_slices:
            average_greyvalues.append(np.mean(im[s]))

    # If you are saving the full frames for later, do that now
    if experiment.save_full_frames:
        filename = output_dir / f"image{str(i).zfill(4)}.tif"
        io.imsave(filename, im)
        
    # If the frame is in the range for the reference frame, ensure it is to be returned
    ref_frame = im if experiment.save_reference_frame and (i in experiment.reference_frame_range) else None

    return (i, average_greyvalues, ref_frame)

# This is the main thread for the parallel processing
if __name__ == '__main__':
    # Initiate a start timer
    total_start_time = time.time()
    # Preallocate some variables
    all_greyvalues = [None] * len(filenames)
    reference_frame_sum = None
    reference_frame_count = 0

    # Initiate a progress bar
    num_files = len(filenames)
    progress_checkpoints = {int(num_files * i / 10) for i in range(1, 11)}
    completed = 0
    
    # Generate the normalisation slice
    calib_slice = roi_to_slice(*calib_topleft)

    # Generate the slices if using an ROI list
    if not experiment.roi_grid:
        roi_slices = [roi_to_slice(*roi) for roi in roi_groups[run.roi_list]]
    if len(standard_rois) > 0:
        std_slices = [roi_to_slice(*roi) for roi in standard_rois]

    # Use ProcessPoolExecutor to perform the parallel processing
    with ProcessPoolExecutor() as executor:
        # This is a dictionary of Future objects indexed by frame number
        # so that we can ensure the results are returned in the correct order
        # at the end
        # This also calls the process_frame_and_save function on each frame
        futures = {
            executor.submit(process_frame_and_save, i, frame): i
            for i, frame in enumerate(filenames)
        }

        # This loop processes each result as it completes (out of order)
        for future in as_completed(futures):
            i, greyvalues, ref_frame = future.result()
            # Update the appropriate element/frame of the grey values list
            all_greyvalues[i] = greyvalues

            # If the result is part of the reference frame then add it to the 
            # reference frame sum
            if ref_frame is not None:
                if reference_frame_sum is None:
                    reference_frame_sum = ref_frame.astype(np.float64)
                else:
                    reference_frame_sum += ref_frame
                reference_frame_count += 1

            # Update the UI with a note every 10%
            completed += 1
            if completed in progress_checkpoints:
                percent = int((completed / num_files) * 100)
                print(f"Processed {percent}% of {num_files} images. ({int(time.time() - total_start_time)} s elapsed)")

    # If saving a reference frame, create the mean now
    if experiment.save_reference_frame and reference_frame_count > 0:
        average_reference_frame = reference_frame_sum / reference_frame_count


# Output a figure with the smoothed reference image and the ROIs marked as
# coloured squares
if experiment.save_reference_frame:
    print("---")
    print("Generating reference image with ROI labels...")
    print(" ")
    reference_frame = average_reference_frame
    # Save one version that is simply the mean, but otherwise untreated so it can be used
    # for subtraction/difference animations, etc.
    io.imsave(output_dir / "reference_image.tif", reference_frame.astype(np.float32, copy=False))
    fig, ax = plt.subplots(figsize=(20,20))
    ax.imshow(reference_frame,
              cmap='gray',
              vmin=np.min(reference_frame),
              vmax=np.percentile(reference_frame,99)    # We use the 99th percentile as a max intensity to crop clipped pixels
              )
    plt.axis("off")
    # Save the figure once without ROIs marked
    fig.savefig(output_dir / "reference_image_noROI.png", bbox_inches="tight")
    # Add the ROIs, whether that's a grid or specific ROIs from the list
    if experiment.roi_grid:
        for row in roi_groups:
            for col in row:
                rect = patches.Rectangle((col[0], col[1]), col[2], col[3],
                                         linewidth = 2,
                                         edgecolor = 'black',
                                         facecolor="none")
                ax.add_patch(rect)
    else:
        for count, roi in enumerate(roi_groups[run.roi_list], start=0):
            rect = patches.Rectangle((roi[0], roi[1]), roi[2], roi[3],
                                     linewidth = 6,
                                     edgecolor = colours[count],
                                     facecolor="none")
            ax.add_patch(rect)
    # Add a patch for the calibration/ROI normalisation region
    rect = patches.Rectangle((calib_topleft[0], calib_topleft[1]), calib_topleft[2], calib_topleft[3],
                             linewidth = 4,
                             edgecolor = "grey",
                             facecolor="none")
    ax.add_patch(rect)
    # Add any standard/quantification ROIs that may be in the image
    if len(standard_rois) > 0:
        for count, roi in enumerate(standard_rois, start=0):
            rect = patches.Rectangle((roi[0], roi[1]), roi[2], roi[3],
                                     linewidth = 6,
                                     edgecolor = "white",
                                     facecolor="none")
            ax.add_patch(rect)
    # Add a scale bar if desired
    if experiment.use_scale_bar:
        scalebar = ScaleBar(experiment.scale_bar[0],
                            "mm",
                            fixed_units = "mm",
                            fixed_value = experiment.scale_bar[1],
                            location = experiment.scale_bar_pos[0],
                            color = experiment.scale_bar_pos[1],
                            box_alpha = 0,
                            font_properties = {"size": 30},
                            border_pad = 1,
                            )
        ax.add_artist(scalebar)
    # Save the figure again with ROIs marked
    plt.show()
    fig.savefig(output_dir / "reference_image_ROI.png", bbox_inches="tight")


# Output a .csv of the average greyvalues for analysis elsewhere
print("---")
print("Saving .csv results for future reference...")
print(" ")
# Create dataframes for all the data we want to save
df = pd.DataFrame(all_greyvalues)
df['Absolute Time'] = times
df['Elapsed Time'] = timestamp
# Generate useful titles for the columns in the dataframe to be saved
columns_titles = ['Absolute Time', 'Elapsed Time']
if experiment.roi_grid:
    for r, row in enumerate(roi_groups[run.roi_list]):
        for c, col in enumerate(row):
            columns_titles.append(r * experiment.grid_cols + c)
    total_rois = experiment.grid_cols * experiment.grid_rows
else:
    for count, roi in enumerate(roi_groups[run.roi_list]):
        columns_titles.append(count)
    total_rois = len(roi_groups[run.roi_list])
if len(standard_rois) > 0:
    for count, roi in enumerate(standard_rois):
        columns_titles.append(total_rois + count)
df = df[columns_titles]
# Rename the columns using the generated column titles
if experiment.roi_grid:
    for r, row in enumerate(roi_groups[run.roi_list]):
        for c, col in enumerate(row):
            df = df.rename(columns={(r * experiment.grid_cols) + c : str(r) + "," + str(c)})
else:
    for count, roi in enumerate(roi_groups[run.roi_list]):
        df = df.rename(columns={count : "ROI " + str(count+1)})
if len(standard_rois) > 0:
    for count, roi in enumerate(standard_rois):
        df = df.rename(columns={total_rois+count : "Standard " + str(count+1)})
        
# Save the final results to a .csv to reference with a later analysis script
out_name = output_dir / f"{run.output_name}_greyvalues.csv"
df.to_csv(out_name, index=False)

# Timestamp the end of the process
total_run_time = time.time() - total_start_time
print(" ")
print("--- Processing finished in %s seconds ---" % (int(total_run_time)))
(output_dir / 'image_processing_timer.txt').write_text(f"Total run time = {total_run_time} s")
