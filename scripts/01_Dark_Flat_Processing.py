"""
A script to take raw dark frames and flat/open beam frames, remove outliers,
and then generate "master dark" and "dark subtracted master flat" from the results.

NB. The master flat output will have already had the dark subtracted.

To-do:
    - Make parallel like the main image processing script?
    
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

import time, json
import numpy as np
from skimage import io
from scipy import ndimage
from dataclasses import asdict
from pathlib import Path

# Import the variables from the experiment config file
# Make sure to add the folder with this file to the sys.path and that no other
# paths to config files are present!
from cfg import experiment

# Timestamp the start of the process
total_start_time = time.time()

# If the output path doesn't exist, create it
output_dir = Path(experiment.exp_outputs_path) / experiment.dark_flat_output_path
if not output_dir.exists():
    print("---")
    print("Output directory doesn't exist, creating it.")
    print(" ")
    output_dir.mkdir(parents=True, exist_ok=True)
    
# Write the settings we used to a .txt for later reference
print("---")
print("Copying parameters and script to files for reference...")
print(" ")
snippet = Path(__file__).read_text()
(output_dir / "dark_flat_processing_script.txt").write_text(snippet)
exp_dict = asdict(experiment)
exp_dict["reference_frame_range"] = list(experiment.reference_frame_range)
(output_dir / "experiment_parameters.txt").write_text(json.dumps(exp_dict, indent=4))

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

# Load the images
print("---")
print("Processing frames...")
print(" ")
for a, paths in enumerate([experiment.dark_paths, experiment.flat_paths]):
    if a == 0:
        print("Processing dark frames")
        print(" ")
    elif a == 1:
        print("Processing flat frames")
        print(" ")
    else:
        raise Exception("More than just darks and flats were given in the script!")
    filenames = []
    for path in paths:
        for file in Path(path).glob('*.tif'):
            filenames.append(file)
    filenames = sorted(filenames)
    
    # For testing. Comment out when running a full batch of calibration frames
    # filenames = filenames[:10]
    
    loading_progress_range = range(0, len(filenames), max(1, int(len(filenames)/10)))
    for i, filename in enumerate(filenames):
        im=io.imread(str(filename))
        
        # Remove outliers if desired
        if experiment.remove_outliers:
            im = outlier_removal(im, 
                                 experiment.bright_threshold, 
                                 experiment.dark_threshold, 
                                 experiment.outlier_radius
                                 )
            
        # If prcessing flats then subtract the previously calculated
        # master dark from each flat before averaging
        if a == 1:
            try: 
                master_dark
            except NameError:
                raise RuntimeError("No master dark computed before flats. Check dark frame paths.")
            im = im - master_dark

        # Initialise the running mean or update it
        if i == 0:
            running_mean = im
        else:
            running_mean = ((i * running_mean) + im) / (i + 1)
            
        # Output some progress as the processing progresses
        if i in loading_progress_range:
            percent = int(round(i/len(filenames)*100, -1))
            e_time = int(time.time() - total_start_time)
            if experiment.remove_outliers:
                print(f"Loaded and removed outliers in {percent}% of {len(filenames)} images. ({e_time} s)")
            else:
                print(f"Loaded {percent}% of {len(filenames)} images.")
    
    print(" ")
    
    if a == 0:
        master_dark = running_mean
        master_dark_numbers = len(filenames)
        print(f"Generating master dark from {master_dark_numbers} frames.")
        print(" ")
        io.imsave((output_dir / "master_dark.tif").as_posix(), 
                  master_dark.astype(np.float32, copy=False)
                  )
        print("Successfully saved master dark to:")
        print((output_dir /  "master_dark.tif").as_posix())
        print(" ")
    elif a == 1:
        master_flat = running_mean
        master_flat_numbers = len(filenames)
        print("Generating master flat from {master_flat_numbers} frames.")
        print(" ")
        io.imsave((output_dir / "dark_subtracted_master_flat.tif").as_posix(), 
                  master_flat.astype(np.float32, copy=False)
                  )
        print("Successfully saved dark subtracted master flat to:")
        print((output_dir / "dark_subtracted_master_flat.tif").as_posix())
        print(" ")
    else:
        raise Exception("More than just darks and flats were given in the script!")

# Timestamp the end of the process
total_run_time = time.time() - total_start_time
print(" ")
print("--- Processing finished in %s seconds ---" % (int(total_run_time)))
with (output_dir / 'dark_flat_settings.txt').open('a') as f:
    f.write("number of dark frames used = " + str(master_dark_numbers) + "\n")
    f.write("number of flat frames used = " + str(master_flat_numbers) + "\n")
    f.write("total_run_time = " + str(total_run_time) + "\n")
