"""
Utility functions used in other scripts in this project.

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
from scipy import ndimage
import numpy as np

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

def create_outputdir(output_dir):
    """
    Create the output directory if it doesn't already exist
    """
    if not output_dir.exists():
        print("---")
        print("Output directory doesn't exist, creating it.")
        print(" ")
        output_dir.mkdir(parents=True, exist_ok=True)

def save_runtime(output_dir, filename, run_time):
    print(" ")
    print("--- Processing finished in %s seconds ---" % (int(run_time)))
    (output_dir / filename).write_text(f"Total run time = {run_time} s")