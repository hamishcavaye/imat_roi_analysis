"""
A configuration script to provide IMAT experiment variables to use when 
processing image data and plotting resulting figures for reports and
publication.

This script defines the variables for one experiment.

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

from class_defs import ExperimentConfig, RunConfig, FigPars

# Provide the experimental variables in the ExperimentConfig dataclass
experiment = ExperimentConfig(
    experiment_name = "IMAT4",
    
    raw_data_path = "/data/instrument/IMAT/2024/RB2420048-1/RB2420048/Cell_Pd_on_Alumina/",
    exp_outputs_path = "/data/analysis/IMAT/RBNumber/RB2420048/Cell_Pd_on_Alumina/Paper Analysis/",
    
    dark_paths = ['/data/instrument/IMAT/2024/RB2420048-1/RB2420048/Cell_Pd_on_Alumina/PH80_Dark2_Start_5sec/', 
                  '/data/instrument/IMAT/2024/RB2420048-1/RB2420048/Cell_Pd_on_Alumina/PH80_Dark_End_5sec/'],
    flat_paths = ['/data/instrument/IMAT/2024/RB2420048-1/RB2420048/Cell_Pd_on_Alumina/PH80_OpenBeam_Start_5sec/', 
                  '/data/instrument/IMAT/2024/RB2420048-1/RB2420048/Cell_Pd_on_Alumina/PH80_OpenBeam_End_5sec/'],
    dark_flat_output_path = "Master_Frames/",
    
    csv_path = "/data/analysis/IMAT/RBNumber/RB2420048/Cell_Pd_on_Alumina/",
    
    use_temp_data = True,
    temp_data_path = "/data/analysis/IMAT/RBNumber/RB2420048/Cell_Pd_on_Alumina/Temp_Data/Temperature_Data_From_Logs.csv",
    temp_data_output_path = "/data/analysis/IMAT/RBNumber/RB2420048/Cell_Pd_on_Alumina/Paper Analysis/Temp Data/",
    temp_data_output_file = "temp_data_pickle",
    
    use_mass_spec_data = True,
    mass_spec_data_path = "/data/analysis/IMAT/RBNumber/RB2420048/Cell_Pd_on_Alumina/Mass_Spec_Data/",
    mass_spec_data = ["2024_12_13 - 1 - Initial He Flow Pre-beam to test MS_modified.csv",
                      "2024_12_13 - 2 - Initial H2 Flow for breakthrough time measurement - Flow inlet-outlet wrong way around_modified.csv",
                      "2024_12_13 - 3 - H2 Flow at 100oC - Catalyst Reduction Hamish.csv",
                      "2024_12_13 - 4 - Acetylene Reaction - Hydrogen Rich.csv",
                      "2024_12_14 - 5 - Acetylene Reaction - Increased T to 100oC.csv",
                      "2024_12_14 - 6 - Acetylene Reaction - H2 Flow Off - 100oC.csv",
                      "2024_12_15 - 7 - Acetylene Reaction - H2 Flow Off - 150oC.csv",
                      "2024_12_15 - 8 - Catalyst H2 Cleaning at 100oC and Ethylene at 60oC.csv",
                      "2024_12_15 - 9 - Ethylene 100oC.csv",
                      "2024_12_16 - 10 - Ethylene 100oC - H2 Off.csv",
                      "2024_12_16 - 11 - Ethylene 150oC - H2 Off.csv"
                      ],
    mass_spec_data_output_path = "/data/analysis/IMAT/RBNumber/RB2420048/Cell_Pd_on_Alumina/Paper Analysis/Mass Spec Data/",
    mass_spec_data_output_file = "mass_spec_data_pickle",
    mass_spec_time_offset = False,
    mass_spec_date_row = 2,
    mass_spec_start_row = 28,
    mass_spec_fieldnames = ['Time', 'ms', 'Hydrogen (2)', 'Helium (4)',
                            'Water (18)', 'Acetylene (26)', 'Ethylene/Ethane (27)',
                            'Ethylene (28)', 'Ethane (30)', 'Argon (40)'],
    
    remove_outliers = True,
    bright_threshold = 1000,
    dark_threshold = 1000,
    outlier_radius = 3,
    
    roi_grid = False,
    roi_groups = [[(850, 800, 330, 60),
                 (850, 890, 330, 60),
                 (850, 980, 330, 60),
                 (850, 1070, 330, 60),
                 (850, 1160, 330, 60),
                 (900, 655, 220, 35),
                 (600, 240, 330, 80)
                 ],
                  [(850, 745, 200, 30),
                   (850, 800, 330, 60),
                   (850, 890, 330, 60),
                   (850, 980, 330, 60),
                   (850, 1070, 330, 60),
                   (850, 1160, 330, 60),
                   (900, 655, 220, 35),
                   (600, 240, 330, 80)
                   ]
                  ],
    calib_topleft = (500, 750, 170, 400),
    standard_rois = [(290, 1070, 100, 250),
                     (290, 1450, 100, 250)
                     ],
    
    colours = ["black",
               "blue",
               "green",
               "magenta",
               "orange",
               "red",
               "gold",
               "navy",
               "white",
               "grey",
               ],
    
    save_full_frames = False,
    
    save_reference_frame = True,
    reference_frame_range = range(0,20),
    use_scale_bar = True,
    scale_bar = (0.103, 50),         # "(pixel scale in mm, scale bar size in mm)"
    scale_bar_pos = ("lower right", "black")
    
)


# Provide individual RunConfig dataclass objects for each run in the experiment
Run1 = RunConfig(
    data_subdir = "PH80_CatalystActivation/",
    csvlog = "IMAT00032939_Cell_Pd_on_AluminaPH80_CatalystActivation_log.csv",
    roi_list = 1,
    output_name = "32939_PH80_CatalystActivation",
    )

Run2 = RunConfig(
    data_subdir = "PH80_Acetylene_HydrogenRich/",
    csvlog = "IMAT00032940_Cell_Pd_on_AluminaPH80_Acetylene_HydrogenRich_log.csv",
    roi_list = 1,
    output_name = "32940_PH80_Acetylene_HydrogenRich",
    )

Run3 = RunConfig(
    data_subdir = "PH80_Acetylene_Deactivation_HighT/",
    csvlog = "IMAT00032941_Cell_Pd_on_AluminaPH80_Acetylene_Deactivation_HighT_log.csv",
    truncate_image_list = [0,4476],
    roi_list = 1,
    output_name = "32941_PH80_Acetylene_Deactivation_HighT",
    )

Run4 = RunConfig(
    data_subdir = "PH80_Acetylene_H2_Flow_Off/",
    csvlog = "IMAT00032943_Cell_Pd_on_AluminaPH80_Acetylene_H2_Flow_Off_log.csv",
    roi_list = 1,
    output_name = "32943_PH80_Acetylene_H2_Flow_Off",
    )

Run5 = RunConfig(
    data_subdir = "PH80_Acetylene_H2_Flow_Off_restart/",
    csvlog = "IMAT00032945_Cell_Pd_on_AluminaPH80_Acetylene_H2_Flow_Off_restart_log.csv",
    roi_list = 1,
    output_name = "32945_PH80_Acetylene_H2_Flow_Off_restart",
    )

Run6 = RunConfig(
    data_subdir = "PH80_Acetylene_H2_Flow_Off_150C/",
    csvlog = "IMAT00032946_Cell_Pd_on_AluminaPH80_Acetylene_H2_Flow_Off_150C_log.csv",
    roi_list = 1,
    output_name = "32946_PH80_Acetylene_H2_Flow_Off_150C",
    )

Run7 = RunConfig(
    data_subdir = "PH80_H2_Cleaning_100C/",
    csvlog = "IMAT00032947_Cell_Pd_on_AluminaPH80_H2_Cleaning_100C_log.csv",
    roi_list = 1,
    output_name = "32947_PH80_H2_Cleaning_100C",
    )

Run8 = RunConfig(
    data_subdir = "PH80_Ethylene_100C/",
    csvlog = "IMAT00032948_Cell_Pd_on_AluminaPH80_Ethylene_100C_log.csv",
    roi_list = 1,
    output_name = "32948_PH80_Ethylene_100C",
    )

Run9 = RunConfig(
    data_subdir = "PH80_Ethylene_H2Off_100C/",
    csvlog = "IMAT00032949_Cell_Pd_on_AluminaPH80_Ethylene_H2Off_100C_log.csv",
    roi_list = 1,
    output_name = "32949_PH80_Ethylene_H2Off_100C",
    )

Run10 = RunConfig(
    data_subdir = "PH80_Ethylene_H2Off_150C/",
    csvlog = "IMAT00032950_Cell_Pd_on_AluminaPH80_Ethylene_H2Off_150C_log.csv",
    roi_list = 1,
    output_name = "32950_PH80_Ethylene_H2Off_150C",
    )

# Generate a list of runs to pass to the analysis scripts
# The first element is "None" simply to allow a "count from 1" capability
# when choosing which run to analyse
run_data = [None,
            Run1,
            Run2,
            Run3,
            Run4,
            Run5,
            Run6,
            Run7,
            Run8,
            Run9,
            Run10,
            ]

# Provide a list of figure dataclasses used for plotting figures
figure_1 = FigPars(
    output_subdir = "Analysis Outputs/Figure 1/",
    fig_name = "Figure 1",
    plot_mode = "both",
    savgol_window = 21,
    savgol_polyorder = 3,
    crop_time = True,
    start_time = "2024-12-13 15:00:00",
    end_time = "2024-12-17 08:00:00",
    mass_spec_pars = [('Hydrogen (2)', 0, 'below', 'Hydrogen'),
                      ('Helium (4)', 1, 'above', 'Helium'),
                      ('Water (18)', 2, 'below', 'Water'),
                      # ('Acetylene (26)', 3, 'below', 'Acetylene'),
                      # ('Ethylene/Ethane (27)', 4, 'below', 'Ethene (27)'),
                      # ('Ethylene (28)', 5, 'below', 'Ethene (28)'),
                      # ('Ethane (30)', 6, 'below', 'Ethane'),
                      ('Argon (40)', 7, 'below', 'Argon'),
                      ],
    plot_temp = True,
    # smooth_temp = True,
    plot_inset = True,
    inset_axes_loc = [0.3, 0.1, 0.4, 0.5],
    inset_axes_xlim = [6,22],
    inset_axes_ylim = [0.99,1.01],
    # plot_arrow = [[1000, "", "up", 0.2, 0.2],
    #               [1500, "2", "down", 0.01, 0.1],
    #               ]
    subplot_labels = [["ms", "a", (0.95, 0.025)],
                      ["n", "b", (0.95, 0.025)],
                      ["inset", "c", (0.05, 0.05)],
                      ],
    subplot_label_size = 120,
    # double_figure_size = (12,8),
    plot_line_width = 12,
    axes_label_font_size = 82,
)

# Generate a list of figures you would like to plot, with their
# respective parameters
figs_to_plot = [figure_1,
                ]