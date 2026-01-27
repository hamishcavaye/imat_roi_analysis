"""
A script to take the saved temperature data from an IMAT log file, process it,
and save it in a pickled format suitable to load in other scripts to plot
against IMAT data.

This script is designed to take the output from JournalViewer when exported
as a .csv

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

import pandas as pd
import json
from dataclasses import asdict
from pathlib import Path

# Import the variables from the experiment config file
# Make sure to add the folder with this file to the sys.path and that no other
# paths to config files are present!
from cfg import experiment

# If the output path doesn't exist, create it
output_dir = Path(experiment.temp_data_output_path)
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
(output_dir / "temp_processing_script.txt").write_text(snippet)
exp_dict = asdict(experiment)
exp_dict["reference_frame_range"] = list(experiment.reference_frame_range)
(output_dir / "experiment_parameters.txt").write_text(json.dumps(exp_dict, indent=4))

# Read the temperature data to a dataframe    
df = pd.read_csv(experiment.temp_data_path, sep=" ",
                 skipinitialspace=True,
                 usecols = [0,1,2],
                 skiprows = [0,1],
                 header=None)

# Reorder the columns and provide names for each series
df = df[[2,0,1]].rename(columns={2 : "Absolute Time", 0 : "Elapsed Time", 1 : "Temperature"})

# Convert the absolute times to datetime format and drop the elapsed time series
df['Absolute Time'] = pd.to_datetime(df['Absolute Time'], format="%d/%m/%yT%H:%M:%S")
df = df.drop('Elapsed Time', axis=1)

# Ensure the data are sorted by absolute time
df.sort_values('Absolute Time', inplace=True)

# Plot the data for the user to check
df.plot('Absolute Time', logy=False)

# Save the output as a pickled object for later use
out_file = output_dir / f"{experiment.temp_data_output_file}.pkl"
df.to_pickle(out_file)