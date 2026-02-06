"""
A script to take calibrated data frames and obtain a set of difference images
between them and the reference frame. 

The resulting images then have a time stamp added and an optional scale bar
before they are saved. 

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
from skimage import io, exposure, restoration
from natsort import natsorted 
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor, as_completed
from matplotlib_scalebar.scalebar import ScaleBar
from dataclasses import asdict
from pathlib import Path

# Import the variables from the experiment config file
# Make sure to add the folder with this file to the sys.path and that no other
# paths to config files are present!
from cfg import experiment, run_data, diff_imgs

# Set which run to analyse
run = 1
run = run_data[run]

img = 0
diffimg = diff_imgs[img]


###############################################################################

output_dir = Path(experiment.exp_outputs_path) / diffimg.output_subdir

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
(output_dir / "diff_img_parameters.txt").write_text(json.dumps(asdict(diffimg), indent=4))


# Timestamp the start of the process
total_start_time = time.time()

# Get the reference frame filename
reference_frame = diffimg.ref_image

# Get the full list of filenames and truncate if necessary
print("---")
print("Obtaining filenames for processing...")
print(" ")
datapath = Path(experiment.exp_outputs_path) / run.output_name
filenames = Path(datapath).glob('*.tif')
# filenames = filenames.remove(reference_frame)
filenames = natsorted(filenames)

print("{} files have been found for processing.".format(str(len(filenames))))
print(" ")

# Extract the frame times from the CSV logs
print("---")
print("Extracting timestamps from log file...")
print(" ")
csv_dir = Path(experiment.exp_outputs_path) / run.output_name
csv_file = run.output_name + "_greyvalues.csv"
csvlog = csv_dir / csv_file
with open(csvlog, newline='') as csvfile:
    reader = csv.DictReader(csvfile,
                            delimiter=',',
                            # fieldnames = ['abstime', 'elapsedtime']
                            )
    timestamp = []  # A list of the absolute times for each frame
    for row in reader:
        timestamp.append(int(float(row['Elapsed Time'])))

if diffimg.frame_select:
    frames = diffimg.frame_select
else:
    frames = (0, len(timestamp))

# Concatenate the filenames and the timestamp lists as appropriate
timestamp = np.array(timestamp) - timestamp[frames[0]]
timestamp = timestamp[frames[0]:frames[1]]
filenames = filenames[frames[0]:frames[1]]

if diffimg.conv_mins:
    timestamp = timestamp/60

# Load the reference frame for subtraction
# then crop it if working with cropped images
# _s is a saved global slice to be used with the parallel processing
ref_img = io.imread(reference_frame)
if diffimg.crop_limits:
    x = diffimg.crop_limits[0][0]
    y = diffimg.crop_limits[0][1]
    w = diffimg.crop_limits[1][0]
    h = diffimg.crop_limits[1][1]
    _s = slice(int(y), int(y+h)), slice(int(x), int(x+w))
    ref_img = ref_img[_s]

# Sequentially go through each file and process it
print("---")
print("Processing frames...")
print(" ")

def pick_scalebar_colour_local(gray_img, location="lower right",
                              patch_frac=0.08,  # fraction of width/height to sample
                              thresh=0.55):
    """
    Returns 'black' or 'white' based on local brightness near `location`.
    gray_img: 2D float array in [0,1]
    location: one of 'lower right', 'lower left', 'upper right', 'upper left'
    patch_frac: size of the sampled patch (as fraction of image dims)
    thresh: brightness threshold to switch colours
    """
    H, W = gray_img.shape[:2]
    ph, pw = max(1, int(H * patch_frac)), max(1, int(W * patch_frac))

    loc = location.lower().strip()
    if "lower" in loc:
        y0, y1 = H - ph, H
    else:  # 'upper'
        y0, y1 = 0, ph

    if "right" in loc:
        x0, x1 = W - pw, W
    else:  # 'left'
        x0, x1 = 0, pw

    patch = gray_img[y0:y1, x0:x1]
    
    # --- NORMALISE LOCALLY ---
    pmin = float(np.min(patch))
    pmax = float(np.max(patch))
    if pmax - pmin < 1e-9:
        # completely flat patch
        b_norm = 0.0 if pmin < 0.5 else 1.0
    else:
        b_norm = (np.median(patch) - pmin) / (pmax - pmin)

    return "white" if b_norm < thresh else "black"

# This is the function that is called in the parallel processing for each frame
# It is passed the frame number (from enumerate(filenames)) and the frame itself
# It uses a number of "global" variables
def process_frame_and_save(i, frame, time_label):
    im = io.imread(frame)
    
    if diffimg.crop_limits:
        im = im[_s]
    
    # simply subtract the reference frame from the current frame
    # diff = util.compare_images(ref_img, im, method='diff')        # This is a built in difference function but it is worse than the subtraction
    diff = im - ref_img
    diff = exposure.rescale_intensity(diff, in_range='image', out_range='float32')     # This rescales the difference image to an appropriate range
    
    # Perform histogram stretch of the 1-99 percentile pixels.
    # This can cause issues if there's dramatic changes of intensity throughout the dataset
    # but works well otherwise to stretch it
    p_lower = np.percentile(diff,diffimg.stretch_pct[0])
    p_upper = np.percentile(diff,diffimg.stretch_pct[1])
    diff = exposure.rescale_intensity(diff, in_range=(p_lower, p_upper))
    
    # Perform some noise reduction if desired (not well tested)
    if diffimg.noise_control:
        diff = restoration.denoise_bilateral(diff)

    # Get the height and width of the image to correctly size the figure/plot
    # and choose DPI    
    height, width = diff.shape[:2]
    dpi = 300
    
    # Plot the figure
    fig = plt.figure(figsize=(width / dpi, height / dpi), dpi=dpi)
    ax = fig.add_axes([0, 0, 1, 1])     # Fills the figure with the axes
    ax.imshow(diff, cmap='gray')        # Plots the image
    ax.axis('off')                      # Turns off axis labels , etc.
    
    # Add the text time stamp
    unit = "min" if diffimg.conv_mins else "s"
    time_label = f"{time_label:.0f} {unit}"
    fig.text(*diffimg.timestamp_loc,
             time_label,
             fontsize = diffimg.timestamp_font_size,
             color = diffimg.timestamp_colour,
             va="top")
    
    # Add the scale bar
    if diffimg.scale_bar:
        loc = diffimg.scale_bar_pos[0]
        bar_colour = pick_scalebar_colour_local(diff, location=loc, patch_frac=0.2, thresh=0.4)
        scalebar = ScaleBar(diffimg.scale_bar[0],
                            "mm",
                            fixed_units = "mm",
                            fixed_value = diffimg.scale_bar[1],
                            location = loc,
                            # location = diffimg.scale_bar_pos[0],
                            # color = diffimg.scale_bar_pos[1],
                            color = bar_colour,
                            box_alpha = 0,
                            font_properties = {"size": 16},
                            border_pad = 1,
                            )
        ax.add_artist(scalebar)
        
    # Draw the final figure then turn it into an image again
    fig.canvas.draw()
    annotated_img = np.asarray(fig.canvas.renderer.buffer_rgba())                               # Renderer only seems to have RGB option so...
    annotated_img = np.dot(annotated_img[...,:3], [0.2989, 0.5870, 0.1140]).astype(np.uint8)    # ... converts back to greyscale
    
    # Close the figure
    plt.close(fig)
    
    # Save the annotated difference frame
    filename = output_dir / f"image{str(i).zfill(4)}.tif"
    io.imsave(filename, annotated_img)

    return None

# This is the main thread for the parallel processing
if __name__ == '__main__':
    # Initiate a start timer
    total_start_time = time.time()

    # Initiate a progress bar
    num_files = len(filenames)
    progress_checkpoints = {int(num_files * i / 10) for i in range(1, 11)}
    completed = 0

    # Use ProcessPoolExecutor to perform the parallel processing
    with ProcessPoolExecutor() as executor:
        # This is a dictionary of Future objects indexed by frame number
        # so that we can ensure the results are returned in the correct order
        # at the end
        # This also calls the process_frame_and_save function on each frame
        futures = {
            executor.submit(process_frame_and_save, i, frame, timestamp[i]): i
            for i, frame in enumerate(filenames)
        }

        # This loop processes each result as it completes (out of order)
        for future in as_completed(futures):
            # i, greyvalues, ref_frame = future.result()
            # # Update the appropriate element/frame of the grey values list
            # all_greyvalues[i] = greyvalues

            # Update the UI with a note every 10%
            completed += 1
            if completed in progress_checkpoints:
                percent = int((completed / num_files) * 100)
                print(f"Processed {percent}% of {num_files} images. ({int(time.time() - total_start_time)} s elapsed)")

# Timestamp the end of the process
total_run_time = time.time() - total_start_time
print(" ")
print("--- Processing finished in %s seconds ---" % (int(total_run_time)))
filename = 'timestamp_processing_timer.txt'
(output_dir / filename).write_text(f"Total run time = {total_run_time} s")