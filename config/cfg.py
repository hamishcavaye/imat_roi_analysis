"""
A configuration script to provide IMAT experiment variables to use when 
processing image data and plotting resulting figures for reports and
publication.

This script defines the variables for one experiment.

Hamish Cavaye - January 2026

To-do list:
    - 
"""

from class_defs import ExperimentConfig, RunConfig, FigPars, ColourMapFig, DiffImg

# Provide the experimental variables in the ExperimentConfig dataclass
experiment = ExperimentConfig(
    experiment_name = "IMAT4",
    
    raw_data_path = "/data/instrument/IMAT/2024/RB2420048-1/RB2420048/Cell_Pd_on_Alumina/",
    exp_outputs_path = "/data/analysis/IMAT/RBNumber/RB2420048/Cell_Pd_on_Alumina/Paper Analysis/",
    
    dark_paths = ['/data/instrument/IMAT/2024/RB2420048-1/RB2420048/Cell_Pd_on_Alumina/PH80_Dark2_Start_5sec/', 
                  '/data/instrument/IMAT/2024/RB2420048-1/RB2420048/Cell_Pd_on_Alumina/PH80_Dark_End_5sec/'],
    flat_paths = ['/data/instrument/IMAT/2024/RB2420048-1/RB2420048/Cell_Pd_on_Alumina/PH80_OpenBeam_Start_5sec/', 
                  '/data/instrument/IMAT/2024/RB2420048-1/RB2420048/Cell_Pd_on_Alumina/PH80_OpenBeam_End_5sec/'],
    dark_flat_output_path = "Master Frames/",
    
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
    
    # roi_grid = True,
    # grid_position = (830, 700, 370, 560),
    # grid_rows = 20,
    # grid_cols = 8,
    roi_groups = [[(850, 800, 330, 60),
                 (850, 890, 330, 60),
                 (850, 980, 330, 60),
                 (850, 1070, 330, 60),
                 (850, 1160, 330, 60),
                 (900, 655, 220, 35),
                 (600, 240, 330, 80)
                 ],
                  [(850, 745, 200, 40),
                   (850, 800, 330, 80),
                   (850, 890, 330, 80),
                   (850, 980, 330, 80),
                   (850, 1070, 330, 80),
                   (850, 1160, 330, 80),
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
    reference_frame_range = range(0,20), #range(815,835),
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
    # output_name = "32940_PH80_Acetylene_HydrogenRich_fullframes_2",
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
    output_subdir = "Paper Figures/01 Reduction (No Smoothing)/",
    fig_name = "Figure 1",
    plot_mode = "both",
    savgol_window = 21,
    savgol_polyorder = 3,
    crop_time = True,
    start_time = "2024-12-13 16:35:00",
    end_time = "2024-12-13 19:30:00",
    smooth = False,
    mask_ndata = False,
    mask_range = [("2024-12-13 17:10:00", "2024-12-13 17:18:00"),
                  ("2024-12-13 18:05:00", "2024-12-13 18:20:00"),
                  ],
    mass_spec_pars = [('Hydrogen (2)', 0, 'below', 'Hydrogen'),
                      ('Helium (4)', 1, 'above', 'Helium'),
                      ('Water (18)', 2, 'below', 'Water'),
                      # ('Acetylene (26)', 3, 'below', 'Acetylene'),
                      # ('Ethylene/Ethane (27)', 4, 'below', 'Ethene (27)'),
                      # ('Ethylene (28)', 5, 'below', 'Ethene (28)'),
                      # ('Ethane (30)', 6, 'below', 'Ethane'),
                      # ('Argon (40)', 7, 'below', 'Argon'),
                      ],
    mass_spec_label_size = 100,
    plot_temp = True,
    # smooth_temp = True,
    plot_inset = True,
    inset_axes_loc = [0.05, 0.3, 0.25, 0.6],
    inset_axes_xlim = [12,21],
    inset_axes_ylim = [0.985,1.01],
    plot_arrow = [[14, "", "up", 0, 0.1, ["n"]],
                  [14, "", "up", 0.1, 0.1, ["ms"]],
                  [14, "", "down", 0.85, 0.15, ["inset"]],
                  ],
    subplot_labels = [["ms", "a", (0.95, 0.025)],
                      ["n", "b", (0.95, 0.025)],
                      ["inset", "c", (0.05, 0.05)],
                      ],
    subplot_label_size = 120,
    # double_figure_size = (12,8),
    plot_line_width = 12,
    axes_label_font_size = 82,
    # custom_ms_ylimits = (0.0000000001, 0.00001),
    custom_temp_ylimits= (0, 180),
)

figure_1bl = FigPars(
    output_subdir = "Paper Figures/01 Drying BL/",
    fig_name = "Figure 1bl",
    plot_mode = "both",
    savgol_window = 21,
    savgol_polyorder = 3,
    crop_time = True,
    start_time = "2024-12-13 17:20:00",
    end_time = "2024-12-13 19:30:00",
    smooth = False,
    mask_ndata = True,
    mask_range = [("2024-12-13 18:05:00", "2024-12-13 18:20:00"),
                  ],
    mass_spec_pars = [('Hydrogen (2)', 0, 'below', 'Hydrogen'),
                      ('Helium (4)', 1, 'above', 'Helium'),
                      ('Water (18)', 2, 'below', 'Water'),
                      # ('Acetylene (26)', 3, 'below', 'Acetylene'),
                      # ('Ethylene/Ethane (27)', 4, 'below', 'Ethene (27)'),
                      # ('Ethylene (28)', 5, 'below', 'Ethene (28)'),
                      # ('Ethane (30)', 6, 'below', 'Ethane'),
                      # ('Argon (40)', 7, 'below', 'Argon'),
                      ],
    mass_spec_label_size = 100,
    plot_temp = True,
    # smooth_temp = True,
    plot_inset = True,
    inset_axes_loc = [0.6, 0.3, 0.25, 0.6],
    inset_axes_xlim = [5,15],
    inset_axes_ylim = [-1.0,0.12],
    # plot_arrow = [[14, "", "up", 0, 0.1, ["n"]],
    #               [14, "", "up", 0.1, 0.1, ["ms"]],
    #               [14, "", "down", 0.85, 0.15, ["inset"]],
    #               ],
    subplot_labels = [["ms", "a", (0.95, 0.025)],
                      ["n", "b", (0.95, 0.8)],
                      ["inset", "c", (0.05, 0.05)],
                      ],
    subplot_label_size = 120,
    # double_figure_size = (12,8),
    plot_line_width = 12,
    axes_label_font_size = 82,
    # custom_ms_ylimits = (0.0000000001, 0.00001),
    custom_temp_ylimits= (0, 180),
    bl = True,
    bl_axis_title = "H$_2$O conc. (mmol cm$^{-3}$)",
    samp_thickness = 1.3,
    total_xsection = 82.574 * 2 + 4.232,
)

figure_2 = FigPars(
    output_subdir = "Paper Figures/02 Acetylene Introduction",
    fig_name = "Figure 2",
    plot_mode = "both",
    # smooth = False,
    savgol_window = 21,
    savgol_polyorder = 3,
    crop_time = True,
    start_time = "2024-12-13 22:20:00",
    end_time = "2024-12-14 03:31:00",
    mass_spec_pars = [('Hydrogen (2)', 0, 'below', 'Hydrogen'),
                      ('Helium (4)', 1, 'above', 'Helium'),
                      # ('Water (18)', 2, 'below', 'Water'),
                      ('Acetylene (26)', 3, 'above', 'Acetylene (26)'),
                      # ('Ethylene/Ethane (27)', 4, 'below', 'Ethene (27)'),
                      ('N2/CO/C2H4 (28)', 5, 'above', 'Ethene (28)'),
                      ('Ethane (30)', 6, 'below', 'Ethane (30)'),
                      ('Argon (40)', 7, 'below', 'Argon'),
                      ],
    plot_temp = True,
    # smooth_temp = True,
    plot_inset = False,
    inset_axes_loc = [0.02, 0.25, 0.2, 0.7],
    inset_axes_xlim = [4,25],
    inset_axes_ylim = [0.985,1.015],
    plot_arrow = [[12, "", "up", 0.8, 0.1, ["n"]],
                  [12, "", "down", 0.87, 0.1, ["ms"]],
                  # [12, "", "down", 0.85, 0.15, ["inset"]],
                  ],
    subplot_labels = [["ms", "a", (0.95, 0.15)],
                      ["n", "b", (0.95, 0.1)],
                      ["inset", "c", (0.05, 0.05)],
                      ],
    subplot_label_size = 120,
    # double_figure_size = (12,8),
    plot_line_width = 12,
    axes_label_font_size = 82,
    custom_temp_ylimits=(0,100),
    custom_ms_ylimits = (0.00000000001, 0.00005),
)

figure_3 = FigPars(
    output_subdir = "Paper Figures/03 Acetylene High T",
    fig_name = "Figure 3",
    plot_mode = "both",
    # smooth = False,
    savgol_window = 21,
    savgol_polyorder = 3,
    crop_time = True,
    start_time = "2024-12-14 11:10:00",
    end_time = "2024-12-15 03:40:00",
    mask_ndata = True,
    mask_range = [("2024-12-14 22:00:00", "2024-12-14 22:40:00"),
                  ],
    mass_spec_pars = [('Hydrogen (2)', 0, 'below', 'Hydrogen'),
                      # ('Helium (4)', 1, 'above', 'Helium'),
                      # ('Water (18)', 2, 'below', 'Water'),
                      ('Acetylene (26)', 3, 'above', 'Acetylene (26)'),
                      # ('Ethylene/Ethane (27)', 4, 'below', 'Ethene (27)'),
                      ('N2/CO/C2H4 (28)', 5, 'above', 'Ethene (28)'),
                      ('Ethane (30)', 6, 'below', 'Ethane (30)'),
                      ('Argon (40)', 7, 'below', 'Argon'),
                      ],
    plot_temp = True,
    # smooth_temp = True,
    plot_inset = True,
    inset_axes_loc = [0.20, 0.07, 0.3, 0.38],
    inset_axes_xlim = [15,60],
    inset_axes_ylim = [0.987,1.008],
    # plot_arrow = [[12, "", "up", 0.8, 0.1, ["n"]],
    #               [12, "", "down", 0.87, 0.1, ["ms"]],
    #               [12, "", "down", 0.85, 0.15, ["inset"]],
    #               ],
    subplot_labels = [["ms", "a", (0.95, 0.15)],
                      ["n", "b", (0.95, 0.1)],
                      ["inset", "c", (0.05, 0.05)],
                      ],
    subplot_label_size = 120,
    # double_figure_size = (12,8),
    plot_line_width = 12,
    axes_label_font_size = 82,
    custom_temp_ylimits=(0,140),
    custom_ms_ylimits = (0.00000000001, 0.00005),
)

figure_4 = FigPars(
    output_subdir = "Paper Figures/04 Catalyst Cleaning",
    fig_name = "Figure 4",
    plot_mode = "both",
    # smooth = False,
    savgol_window = 21,
    savgol_polyorder = 3,
    crop_time = True,
    start_time = "2024-12-15 18:20:00",
    end_time = "2024-12-15 20:20:00",
    mask_ndata = True,
    mask_range = [("2024-12-15 19:15:00", "2024-12-15 19:40:00"),
                  ],    mass_spec_pars = [('Hydrogen (2)', 0, 'below', 'Hydrogen'),
                      ('Helium (4)', 1, 'above', 'Helium'),
                      ('Water (18)', 2, 'above', 'Water'),
                      ('Acetylene (26)', 3, 'above', 'Acetylene (26)'),
                      # ('Ethylene/Ethane (27)', 4, 'below', 'Ethene (27)'),
                      ('N2/CO/C2H4 (28)', 5, 'above', 'Ethene (28)'),
                      ('Ethane (30)', 6, 'above', 'Ethane (30)'),
                      ('Argon (40)', 7, 'below', 'Argon'),
                      ],
    plot_temp = True,
    # smooth_temp = True,
    plot_inset = False,
    inset_axes_loc = [0.02, 0.25, 0.2, 0.7],
    inset_axes_xlim = [4,25],
    inset_axes_ylim = [0.985,1.015],
    plot_arrow = [[16, "", "down", 0.65, 0.1, ["n"]],
                  [16, "", "down", 0.87, 0.1, ["ms"]],
                  # [12, "", "down", 0.85, 0.15, ["inset"]],
                  ],
    subplot_labels = [["ms", "a", (0.95, 0.15)],
                      ["n", "b", (0.95, 0.1)],
                      ["inset", "c", (0.05, 0.05)],
                      ],
    subplot_label_size = 120,
    # double_figure_size = (12,8),
    plot_line_width = 12,
    axes_label_font_size = 82,
    custom_temp_ylimits=(0,140),
    custom_ms_ylimits = (0.00000000001, 0.00005),
)

# Generate a list of figures you would like to plot, with their
# respective parameters
figs_to_plot = [None,
                figure_1,
                figure_2,
                figure_3,
                figure_4,
                figure_1bl,
                ]

grid_fig_1 = ColourMapFig(
    run = 1,
    crop_time = True,
    start_time = "2024-12-13 16:50:00",
    end_time = "2024-12-13 17:20:00",
    output_subdir = "Analysis Outputs/CMAP Fig 1 BL/",
    fig_name = "CMAP Fig 1",
    colour_minmax = (0, 0.12),
    ref_image = "/data/analysis/IMAT/RBNumber/RB2420048/Cell_Pd_on_Alumina/Paper Analysis/32939_PH80_CatalystActivation_Grid/reference_image.tif",
    crop_image = True,
    # conv_mins = False,
    # smooth = True,
    bl = True,
    samp_thickness = 1.3,
    total_xsection = 82.574 * 2,
    )

grid_figs = [grid_fig_1]

diff_img_1 = DiffImg(
    run = 2,
    output_subdir = "Paper Figures/Acetylene Diff Fig 1 Late Reference/",
    fig_name = "Acetylene Diff Fig 1 Late Reference",
    ref_image = "/home/hc1089498/Desktop/IMAT 4 (RB2420048)/Cell_Pd_on_Alumina/Paper Analysis/32940_PH80_Acetylene_HydrogenRich_fullframes_2/reference_image.tif",
    crop_limits = ((480, 300), (1100, 1200)),
    frame_select = (885, 3736),
    scale_bar = (0.103, 20),         # "(pixel scale in mm, scale bar size in mm)"
    timestamp_colour = "black",
    
    )

diff_imgs = [None,
             diff_img_1]