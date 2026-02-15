"""
A script that takes previously processed .csv data from a series of neutron 
radiography runs and allows the plotting across runs, with and without various
types of normalisation.

To-do list:
    - Make the text sizes relative to figure size?
    - Make a version for "live" analysis of data during experiments? i.e. no .csv log available
    
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

import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.transforms as transforms
import time, json
from scipy.signal import savgol_filter
from dataclasses import asdict
from pathlib import Path
from utils import create_outputdir, save_runtime

# Import the variables from the experiment config file
# Make sure to add the folder with this file to the sys.path and that no other
# paths to config files are present!
from cfg import experiment, run_data, figs_to_plot

# Choose which figure to plot (counts from 0!)
figpars = figs_to_plot[0]

###############################################################################

# Get all the run .csv files from the config file
input_dir = Path(experiment.exp_outputs_path)
runs = []
for run in run_data:
    if run is not None:
        run_csv = input_dir / run.output_name / f"{run.output_name}_greyvalues.csv"
        runs.append(run_csv)

# Get the start time so we can report total processing time later
print("---")
print("Starting data analysis...")
print(" ")
total_start_time = time.time()

# Import a list of colours to use for the ROIs
colours = experiment.colours

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
(output_dir / "figure_parameters.txt").write_text(json.dumps(asdict(figpars), indent=4))

# Start loading the data
print("---")
print("Loading the previously saved data...")
print(" ")
# Create some empty dicts to keep the dataframes from each run in
runs_df = {}
# Create an empty dataframe to hold the concatenated data
total_data = pd.DataFrame()

for run_no, run in enumerate(runs, start=1):
    # Read the run into a dataframe from the .csv, converting the time from
    # a string to a timestamp data type
    df = pd.read_csv(run)
    df['Absolute Time'] = pd.to_datetime(df['Absolute Time'])
    # Save the dataframe with the run number as a dict key and then 
    # concatenate the run with the total data dataframe
    runs_df.update({"Run "+str(run_no): df})
    total_data = pd.concat([total_data, df], ignore_index=True)

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

print("Successfully loaded "+str(len(runs))+" runs.")
print(" ")

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
        
# Some test plots to see the results
# if plot_test_plots:
#     ROI_list = ['ROI 1', 'ROI 2', 'ROI 3', 'ROI 4', 'ROI 5', 'ROI 6', 'ROI 7']
#     total_data.plot(x ='Elapsed Time', y = ROI_list)
#     total_data_norm.plot(x ='Elapsed Time', y = ROI_list)

# If using mass spec data, load the appropriate data now
if figpars.plot_mode in ["both", "ms_only"]:
    print("Extracting relevant portion of the mass spec data...")
    mass_spec = pd.read_pickle(experiment.mass_spec_data_output_path + experiment.mass_spec_data_output_file + '.pkl')
    filtered_mass_spec = mass_spec.loc[(mass_spec['Absolute Time'] >= total_data['Absolute Time'].iloc[0])
                                        & (mass_spec['Absolute Time'] < total_data['Absolute Time'].iloc[-1])]
    # Convert the absolute times to elapsed time since the start of the neutron data
    abs_time = []
    for tstamp in filtered_mass_spec['Absolute Time']:
        t = tstamp-total_data['Absolute Time'].iloc[0]
        if figpars.conv_mins:
            abs_time.append(t.total_seconds()/60)
        else:
            abs_time.append(t.total_seconds())
    # Add this elapsed time as a column in the pandas dataframe and drop the
    # absolute time column    
    filtered_mass_spec.insert(1, 'Elapsed Time', abs_time)
else:
    filtered_mass_spec = None

# If using temperature data, load the appropriate data now
if figpars.plot_temp:
    print("Extracting relevant portion of the temperature data...")
    temperature = pd.read_pickle(experiment.temp_data_output_path + experiment.temp_data_output_file + '.pkl')
    filtered_temperature = temperature.loc[(temperature['Absolute Time'] >= total_data['Absolute Time'].iloc[0])
                                        & (temperature['Absolute Time'] < total_data['Absolute Time'].iloc[-1])]
    abs_time = []
    # Convert the absolute times to elapsed time since the start of the neutron data
    for tstamp in filtered_temperature['Absolute Time']:
        t = tstamp-total_data['Absolute Time'].iloc[0]
        if figpars.conv_mins:
            abs_time.append(t.total_seconds()/60)
        else:
            abs_time.append(t.total_seconds())
    # Add this elapsed time as a column in the pandas dataframe and drop the
    # absolute time column    
    filtered_temperature.insert(1, 'Elapsed Time', abs_time)
    if figpars.smooth_temp:
        filtered_temperature.loc[:, 'Temperature'] = savgol_filter(filtered_temperature['Temperature'],
                                                                   181,
                                                                   2)
else:
    filtered_temperature = None
            

# Define some functions that will plot the various datasets
def plot_mass_spec(ax, ms_data):
    ax.set_label("ms")
    for count, line in enumerate(ms_data.columns):
        if line != 'Elapsed Time' and line != 'Absolute Time' and line in([frag[0] for frag in figpars.mass_spec_pars]):
            # This little for/if section gets the correct plot colour for the 
            # mass spec line being plotted
            for frag in figpars.mass_spec_pars:
                if line in(frag):
                    col = experiment.colours[frag[1]]
                    label = frag[3]
                    if frag[2] == "above":
                        xy_pos = (-50,10)
                    elif frag[2] == "below":
                        xy_pos = (-50,-130)
            ax.plot(ms_data['Elapsed Time'],
                        ms_data[line],
                        linewidth=figpars.plot_line_width, c=col)
            ax.annotate(label,
                        xy=(ms_data['Elapsed Time'].iloc[-1], ms_data[line].iloc[-1]), 
                        xytext=xy_pos, 
                        horizontalalignment='right', 
                        fontsize=figpars.mass_spec_label_size,
                        xycoords='data', 
                        textcoords='offset pixels',
                        verticalalignment='bottom',
                        annotation_clip = False,
                        c=col)
    # Set the y-axis scale to log
    ax.set_yscale('log')
    # change all spines
    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(4)
    ax.set_ylabel("Partial Pressure (Arb. units)", fontsize=figpars.axes_label_font_size)
    if figpars.custom_ms_ylimits:
        ax.set_ylim(figpars.custom_ms_ylimits[0], figpars.custom_ms_ylimits[1])

def plot_neutron(ax, n_data):     
    if figpars.plot_inset:
        inset_ax = ax.inset_axes(figpars.inset_axes_loc,
                                 xlim=figpars.inset_axes_xlim,
                                 ylim=figpars.inset_axes_ylim,
                                 # xticklabels=[],
                                 yticklabels=[]
                                 )
        inset_ax.set_label("inset")
    ax.set_label("n")
    for count, line in enumerate(n_data.columns):
        if line != 'Elapsed Time' and line != 'Absolute Time' and ("Standard" not in line):
            if figpars.mask_ndata:
                ax.plot(n_data.loc[mask, 'Elapsed Time'], n_data.loc[mask, line], 
                         linewidth=figpars.plot_line_width, c=colours[count-2])
            else:
                ax.plot(n_data['Elapsed Time'], n_data[line], 
                         linewidth=figpars.plot_line_width, c=colours[count-2])
        if figpars.plot_inset:
            inset_ax.plot(n_data['Elapsed Time'], n_data[line],
                        c=colours[count-2], linewidth=figpars.plot_line_width)
    # change all spines
    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(4)
        if figpars.plot_inset:
            inset_ax.spines[axis].set_linewidth(4)
    if figpars.plot_inset:
        ax.indicate_inset_zoom(inset_ax, edgecolor="black", linewidth = 5, alpha = None)
        inset_ax.tick_params(axis='x', labelsize=figpars.tick_font_size, pad=10, length=10, width=3)
        inset_ax.tick_params(axis='y', labelsize=figpars.tick_font_size, pad=10, length=10, width=3)
    if figpars.normalise:
        ax.set_ylabel("Ave. Grey Value (Norm., Arb. units)", fontsize=figpars.axes_label_font_size)
    else:
        ax.set_ylabel("Ave. Grey Value (Arb. units)", fontsize=figpars.axes_label_font_size)
    if figpars.custom_neutron_ylimits:
        ax.set_ylim(figpars.custom_neutron_ylimits[0], figpars.custom_neutron_ylimits[1])

def plot_temp(ax, t_data):
    ax2 = ax.twinx()
    ax2.plot(t_data['Elapsed Time'],
             t_data['Temperature'],
             linewidth = figpars.plot_line_width, c='black', ls = 'dashed')
    ax2.set_ylabel("Temperature (\N{DEGREE SIGN}C)", fontsize=figpars.axes_label_font_size, rotation=270, labelpad=80)  
    if figpars.custom_temp_ylimits:
        ax2.set_ylim(figpars.custom_temp_ylimits[0], figpars.custom_temp_ylimits[1])

# Define a function to plot arrows on the figure with labels at specific points        
def plot_arrow_pct(ax, x_val, label, direction="up", y_pct=0.5, length_pct=0.1):
    # Create a transform: X is data, Y is axis (0-1)
    trans = transforms.blended_transform_factory(ax.transData, ax.transAxes)
    # Determine arrow direction and start/end points
    if direction == "up":
        y_start = y_pct - length_pct
        y_end = y_pct
        va = 'top'
    else:
        y_start = y_pct + length_pct
        y_end = y_pct
        va = 'bottom'
    # Draw the arrow
    ax.annotate(label, xy=(x_val, y_end), xycoords=trans, xytext=(x_val, y_start),
                fontsize=80, ha='center', va=va,
                arrowprops=dict(facecolor='red', width=25, headwidth=50,
                                headlength=50, shrink=0.05),
                transform=trans)

# Define a main function to actually plot the data, using the plot functions
# that are above
def plot_figure(n_data, ms_data=None, t_data=None, mode="both"):
    if mode == "both":
        fig, (ax_top, ax_bottom) = plt.subplots(2, 1, sharex=True, figsize=figpars.double_figure_size, gridspec_kw={'hspace': 0})
        ax_bottom.tick_params(axis='x', top=True, labeltop=False, bottom=True, labelbottom=True)
        plot_mass_spec(ax_top, ms_data)
        plot_neutron(ax_bottom, n_data)
        if t_data is not None:
            plot_temp(ax_top, t_data)
        label_ax = ax_bottom
    elif mode == "ms_only":
        fig, ax_top = plt.subplots(figsize=figpars.figure_size)
        plot_mass_spec(ax_top, ms_data)
        if t_data is not None:
            plot_temp(ax_top, t_data)
        label_ax = ax_top
    elif mode == "neutron_only":
        fig, ax_bottom = plt.subplots(figsize=figpars.figure_size)
        plot_neutron(ax_bottom, n_data)
        label_ax = ax_bottom
    for ax in fig.axes:
        ax.set_xmargin(0)
        ax.tick_params(labelsize=figpars.tick_font_size, pad=10, length=20, width=4)
        if figpars.plot_vline:
            for line in figpars.plot_vline:
                ax.axvline(x = line, color = 'r', linestyle = '--', linewidth = 10)
        if figpars.plot_arrow:
            for arrow in figpars.plot_arrow:
                plot_arrow_pct(ax, arrow[0], arrow[1], direction=arrow[2], y_pct=arrow[3], length_pct=arrow[4])
        if figpars.subplot_labels:
            for label in figpars.subplot_labels:
                if label[0] == ax.get_label():
                    ax.annotate(label[1], xy = label[2], xycoords = "axes fraction", fontsize = figpars.subplot_label_size)
    if figpars.plot_inset and figpars.subplot_labels:
        all_axes = fig.axes
        insets = [child for ax in all_axes for child in ax.get_children() if isinstance(child, plt.Axes)]
        for inset in insets:
            for label in figpars.subplot_labels:
                if label[0] == inset.get_label():
                    inset.annotate(label[1], xy = label[2], xycoords = "axes fraction", fontsize = figpars.subplot_label_size)
                
    x_label = "Time (min)" if figpars.conv_mins else "Time (s)"
    label_ax.set_xlabel(x_label, fontsize=figpars.axes_label_font_size)
    plt.tight_layout()
    plt.draw()
    
    return fig

# Choose the neutron data to plot (normalised or not)
if figpars.normalise:
    data_to_plot = total_data_norm.copy()
else:
    data_to_plot = total_data.copy()

# If you want to omit a region of the neutron data then this will mask a section
if figpars.mask_ndata:
    mask = (data_to_plot["Absolute Time"] < figpars.mask_start) | (data_to_plot["Absolute Time"] > figpars.mask_end)

# Actually call the plot function to plot the desired figure
figure = plot_figure(data_to_plot, filtered_mass_spec, filtered_temperature, figpars.plot_mode)

# Save the figure.
out_file = output_dir / f"{figpars.fig_name}.png"
figure.savefig(out_file, bbox_inches="tight")

# Finally save a file to say how long the total script ran for
total_run_time = time.time() - total_start_time
save_runtime(output_dir, "analysis_script_timer.txt", total_run_time)

