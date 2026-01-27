"""
A script to take the saved output from a Hiden mass spectrometer, process it,
and save it in a pickled format suitable to load in other scripts to plot
against IMAT data.

NB. All .csv files given as inputs must have the same number of mass spec 
traces. They need to be modified manually if there are discrepancies.

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
import csv, json
import datetime
from dataclasses import asdict
from pathlib import Path

# Import the variables from the experiment config file
# Make sure to add the folder with this file to the sys.path and that no other
# paths to config files are present!
from cfg import experiment

# If the output path doesn't exist, create it
output_dir = Path(experiment.mass_spec_data_output_path)
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
(output_dir / "mass_spec_processing_script.txt").write_text(snippet)
exp_dict = asdict(experiment)
exp_dict["reference_frame_range"] = list(experiment.reference_frame_range)
(output_dir / "experiment_parameters.txt").write_text(json.dumps(exp_dict, indent=4))

# Initialise an empty dataframe and list of start times
mass_spec_dataframe = pd.DataFrame()
start_times = []

# Iterate over each file
for file in experiment.mass_spec_data:
    with open(Path(experiment.mass_spec_data_path) / file, newline='') as csvfile:
        reader = csv.DictReader(csvfile,
                                delimiter=',',
                                fieldnames = experiment.mass_spec_fieldnames
                                )
        for number, row in enumerate(reader):
            # Get the start date and start time
            if number == experiment.mass_spec_date_row:
                start_date = row[experiment.mass_spec_fieldnames[1]]
                start_time = row[experiment.mass_spec_fieldnames[3]]
        dformat = "%d/%m/%Y %H:%M:%S"
        start_time_parsed = datetime.datetime.strptime(start_date + " " + start_time, dformat)
        
        # If the mass spec absolute time has an offset from the neutron data
        # it can be applied here.
        if experiment.mass_spec_time_offset != False:
            start_time_parsed_corrected = start_time_parsed - datetime.timedelta(seconds=experiment.mass_spec_time_offset)
            start_times.append(start_time_parsed_corrected)
        else:
            start_times.append(start_time_parsed)
    
    df = pd.read_csv(experiment.mass_spec_data_path+file,
                     header=experiment.mass_spec_start_row)
    
    # Give the user feedback on which file is being processed
    print(file)
    
    # Ininitialise a list for the absolute timestamps
    absolute_timestamps = []
    
    # Iterate over each row in the Time series 
    for tstamp in df['Time']:
        t = pd.Timedelta(tstamp).to_pytimedelta()
        if experiment.mass_spec_time_offset:
            absolute_timestamps.append(start_time_parsed_corrected + t)
        else:
            absolute_timestamps.append(start_time_parsed + t)
        
    # Insert the Absolute Time series to the dataframe
    df.insert(0, 'Absolute Time', absolute_timestamps)
    
    # Concatenate this file with the whole dataset
    mass_spec_dataframe = pd.concat([mass_spec_dataframe, df], ignore_index=True)

# Tidy up the dataframe before saving it    
mass_spec_dataframe = mass_spec_dataframe.loc[:, :experiment.mass_spec_fieldnames[-1]]
mass_spec_dataframe = mass_spec_dataframe.drop(['ms','Time'], axis=1)
mass_spec_dataframe.sort_values('Absolute Time', inplace=True)

# Plot the processed data for checking
mass_spec_dataframe.plot('Absolute Time', logy=True)

# Save the dataframe as a pickled object for later use
out_file = output_dir / f"{experiment.mass_spec_data_output_file}.pkl"
mass_spec_dataframe.to_pickle(out_file)