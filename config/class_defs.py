"""
A configuration script to provide IMAT experiment variables to use when 
processing image data and plotting resulting figures for reports and
publication.

This script defines the dataclasses for importing and use in an experiment's 
configuration file.

Hamish Cavaye - January 2026

To-do list:
    - 
"""

from dataclasses import dataclass, field
from typing import Optional
from matplotlib.colors import Colormap
from matplotlib import colormaps

# A list of colours to use for plots and ROIs to help ensure colours remain 
# consistent throughout different plots
DEFAULT_COLOURS = ["black",
                   "blue",
                   "green",
                   "magenta",
                   "orange",
                   "red",
                   "gold",
                   "navy",
                   "white",
                   "grey",
                   ]

# The default figure label positions and letters if none are specified
DEFAULT_LABELS = [["ms", "a", (0.95, 0.025)],
                  ["n", "b", (0.95, 0.025)],
                  ["inset", "c", (0.05, 0.05)],
                  ]

# Define the dataclasses to use
# One for the overall experiment config and data directories
# One for the individual runs
# Three for plotting different types of figures
@dataclass(kw_only=True)
class ExperimentConfig:
    experiment_name: str                # Used only for internal reference
    
    raw_data_path: str                  # The parent directory containing the raw image data (full path)
    exp_outputs_path: str               # The parent directory for all analysis outputs (full path)
    
    dark_paths: list                    # List of the full paths to all dark frame data to use
    flat_paths: list                    # List of the full paths to all open beam data to use
    dark_flat_output_path: str          # subdirectory under exp_outputs_path to store master calibration frames
    
    csv_path: Optional[str] = None      # Full path to the location of the neutron run logfiles
    
    use_temp_data: bool = False                     # A flag to define whether temperature data is present and to be used
    temp_data_path: Optional[str] = None            # Full path to the temperature .csv data
    temp_data_output_path: Optional[str] = None     # Full path to a directory to save the parsed temperature data
    temp_data_output_file: Optional[str] = None     # Filename to use for the parsed temperature data
    
    use_mass_spec_data: bool = False                    # A flag to define whether mass spec data is present and to be useed
    mass_spec_data_path: Optional[str] = None           # Full path to a directory of mass spec data files
    mass_spec_data: Optional[list] = None               # A list of mass spec data file names to use
    mass_spec_data_output_path: Optional[str] = None    # Full path to a directory to save the parsed mass spec data
    mass_spec_data_output_file: Optional[str] = None    # Filename to use efor the parsed mass spec data
    mass_spec_time_offset: Optional[int] = None         # Integer value to compensate for any offset in the absolute times in the mass spec data and the neutron data (minutes). False if not needed.
    mass_spec_date_row: Optional[int] = None            # The 0 counted row number containing the start date and time
    mass_spec_start_row: Optional[int] = None           # The 0 counted row number containing the data headings "Time" etc.
    mass_spec_fieldnames: Optional[list] = None         # A list of the different mass spec column titles - must be present in all mass spec .csv files
    
    remove_outliers: bool = False               # A flag to define whether outlier removal should take place
    bright_threshold: Optional[int] = None      # The intensity threshold for bright pixel outlier removal
    dark_threshold: Optional[int] = None        # The intensity threshold for dark pixel outlier removal
    outlier_radius: int = 3                     # Pixel radius to use for outlier removal
    
    roi_grid: bool = False              # A boolean flag for whether a grid of ROIs should be used or not
    # Provide the size and shape of the area for the grid in the format
    # (top left x pixel, top left y pixel, width, height)
    grid_position: Optional[tuple[int, int, int, int]] = None
    grid_rows: Optional[int] = None     # Integer number of grid rows to use in a grid of ROIs
    grid_cols: Optional[int] = None     # Integer number of grid columns to use in a grid of ROIs
    # If not using an ROI grid (i.e. roi_grid = False):
    # Give the pixel coordinates of the top-left pixel for each ROI of interest.
    # Multiple ROI groups can be given here (it is a list of lists). 
    # Format is "(top left x pixel, top left y pixel, roi width, roi height)"
    roi_groups: Optional[list[list[tuple[int, int, int, int]]]] = None
    calib_topleft: tuple        # e.g. an air region for ROI normalisation or an internal standard point. Format is "(top left x pixel, top left y pixel, roi width, roi height)"
    standard_rois: Optional[list[tuple[int, int, int, int]]] = None     # An optional list of tuples to define ROIs that cover internal standards in the frame. Same format as the above ROIs.
    
    colours: list[str] = field(default_factory=lambda: DEFAULT_COLOURS.copy())      # A list of colours to use for plotting, defaults are given in this script at the top
    
    save_full_frames: bool = False          # A boolean flag defining whether the complete calibrated frames should be saved when processing a neutron run
    
    save_reference_frame: bool = True                   # A boolean flag to determine whether a low-noise average reference frame should be saved when processing a neutron run
    reference_frame_range: list = range(0,20)           # Which frames (in a list) should be useed to generate the reference frame
    use_scale_bar: bool = False                         # A boolean flag to trigger inclusion of a scalebar in the reference frames
    scale_bar: Optional[tuple[float, int]] = None       # Scalebar parameters as a tuple in the format (pixel scale in mm, scale bar size in mm), None = don't plot a scalebar
    scale_bar_pos: tuple = ("lower right", "black")     # Scalebar position and colour, see https://pypi.org/project/matplotlib-scalebar/
    
@dataclass(kw_only=True)
class RunConfig():
    data_subdir: str                                        # Subdirectory relative to raw_data_path for the image data for the run
    csvlog: Optional[str] = None                            # Filename of the csv log file for the run
    truncate_image_list: Optional[list[int]] = None         # List of frames to use if not None (and therefore using the wholedataset). Often defined as list(range(start, end))
    truncate_log: Optional[list[int]] = None                # As above but for the entries in the csv logfile. Sometimes useful when the log file contains extraneous lines
    roi_list: int = 0                                       # Integer list indeex denoting which ROI group to use (if not using a grid)
    output_name: str                                        # Subdirectory name to store the analysis outputs for the run, under exp_outputs_path
    frame_skip: Optional[int] = None                        # Integer value denoting whether to only process every nth frame, useful for testing things with large datasets
    crop_image: Optional[bool] = False                      # Boolean flag as to whether to crop the calibrated frames when saving full frames and reference frames
    crop_area: Optional[tuple[int, int, int, int]] = None   # Crop area parameters in the format (top left x pixel, top left y pixel, area width, area height)
    
@dataclass(kw_only=True)
class FigPars():
    output_subdir: str                      # A subdirectory to save the figure and corresponding outputs under exp_outputs_path
    fig_name: str                           # A figure name, used as a filename for the saved figure
    plot_mode: str = "neutron_only"         # Whether to plot "neutron_only", "ms_only" or "both"
    conv_mins: bool = True                  # A flag to determine if the time axis should be plotted in minutes. False resorts to plotting in seconds.
    normalise: bool = True                  # A flag to determine if the neutron data should have the first point normalised to 1. NB. This is the first point of the whole experiment, not the cropped time period if one is given
    smooth: bool = True                     # A flag to turn on or off Savitzky-Golay smoothing of the neutron data
    savgol_window: int = 21                 # The Savitzky-Golay smoothing window size
    savgol_polyorder: int = 3               # The Savitzky-Golay smoothing polynomial order
    crop_time: bool = False                 # A flag to determine whether the entire experiment should be plotted (False), or a specific time period (True)
    start_time: Optional[str] = None        # The start date and time for a cropped time plot as a string in format "YYYY-MM-DD HH:MM:SS"
    end_time: Optional[str] = None          # The end date and time for a cropped time plot as a string in format "YYYY-MM-DD HH:MM:SS"
    tzero: bool = True                      # Setting this to True will set the first time point in the plot to be 0
    normalise_tzero: bool = True            # A flag to determine if you want to renormalise the first point in the neutron data to 1 for this cropped time period
    mask_ndata: bool = False                # A flag to determine if you want to mask any of the neutron data, e.g. for missing neutron data from beam losses
    mask_range: list[tuple[str, str]] = field(default_factory=lambda: None)     # A list of tuples defining regions of neutron data to mask. Format of each tuple is ("start datetime", "end datetime") in the same format as start_time and end_time, above
    # If you are plotting mass spec data, you need to define which mass spec ions you want to plot
    # This is a list where each element is one line
    # The format of each list element is a 4-tuple with the format:
    # ('mass spec fragment column name', integer index for line colour, label position 'above' or 'below', 'label to use')
    mass_spec_pars: Optional[list[tuple[str, int, str, str]]] = None
    plot_temp: bool = False                 # A flag to determine whether to plot temperature data on the mass spec axes
    smooth_temp: bool = False               # A flag to determine whether to smooth the temperature data or not
    plot_inset: bool = False                # A flag to enable or disable an inset plot for the neutron data
    # Inset axes location parameters as a 4-element list in the format
    # [x axis fraction for bottom left point, y axis fraction for bottom left point, x axis fraction for plot width, y axis fraction for plot height]
    inset_axes_loc: Optional[list[float, float, float, float]] = None       
    inset_axes_xlim: Optional[list[int, int]] = None                # The x-axis limits for the inset plot as a 2-element list with the format [start, end]
    inset_axes_ylim: Optional[list[float, float]] = None            # The y-axis limits for the inset plot as a 2-element list with the format [lower bound, upper bound]
    custom_ms_ylimits: Optional[tuple[float, float]] = None         # Optional limits for the mass spec y-axis as a 2-tuple in the format (lower bound, upper bound)
    custom_neutron_ylimits: Optional[tuple[float, float]] = None    # Optional limits for the neutron data y-axis as a 2-tuple in the format (lower bound, upper bound)
    custom_temp_ylimits: Optional[tuple[int, int]] = None           # Optional limits for the temperature data y-axis as a 2-tuple in the format (lower bound, upper bound)
    axes_label_font_size: int = 64          # Axis label font size 
    tick_font_size:int  = 64                # Axis tick label font size
    figure_size: tuple = (40, 20)           # Figure size of a single mass spec or neutron plot
    double_figure_size: tuple = (60, 40)    # Figure size for a combined mass spec and neutron plot
    mass_spec_label_size: int = 60          # Font size for the mass spec trace labels
    # A list of arrows to be annotated on the figure. Each list element is one arrow with the format
    # [x-axis position, "text label", "up" or "down", y-axis fraction for the arrow point, y-axis fraction for the arrow length, list of subplots to apply to "n", "ms", "inset"]
    plot_arrow: Optional[list[list[int, str, str, float, float, list[str]]]] = None
    plot_vline: Optional[list[int]] = None  # An optional vertical dashed red line to plot across all axes at the given x-axis position, given as a list for each line
    # Labels to be annotated on subplots for publication, given as a list of lists in the format:
    # ["subplot label", "text label to use", (x-axis fraction for position, y-axis fraction for position)]
    subplot_labels: list[list[str, str, tuple[float, float]]] = field(default_factory=lambda: DEFAULT_LABELS.copy())
    subplot_label_size: int = 80        # Font size for the subplot labels
    plot_line_width: int = 8            # Line width for the lines
    bl: bool = False                    # A flag to determine whether to perform Beer-Lambert conversion of the neutron y-axis units
    bl_axis_title: str = "H$_2$ conc. (mmol cm$^{-3}$)"     # A title to give for the neutron y-axis if Beer-Lambert has been used
    samp_thickness: Optional[float] = None  # Neutron beam path length in cm for Beer-Lambert analysis
    total_xsection: Optional[float] = None  # Total neutron cross section to use for Beer-Lambert analysis, in barns
    
@dataclass(kw_only=True)
class ColourMapFig():
    run: int                # The run this colour map figure relates to, as an index from the run_data list
    output_subdir: str      # The subdirectory to save the outputs, under exp_outputs_path
    fig_name: str           # The name for the figure, used as filenames
    save_format: str = ".png"   # The file format to use for saved images, usually ".png" or ".tif"
    cmap: Colormap = field(default_factory=lambda: colormaps["inferno"])    # Which colourmap to use
    colour_minmax: tuple[float, float] = (0, 1.0)   # The minimum and maximum extent of the colourmap (aka "z-axis limits") as a 2-element tuple
    log: bool = False               # Whether to plot the colour map on a log scale or not
    heatmap_alpha: float = 0.7      # Transparency to use on the heatmap overlay
    bar_shrink_factor: float = 0.9  # Adjusts the size of the heatmap colour legend/bar
    ref_image: str                  # Full path to the reference image filename to use under the colourmap
    conv_mins: bool = True          # A flag to determine if the timestamp should be converted to mins or remain in seconds
    normalise: bool = True          # A flag to determine whether to normalise the transmission data to 1 at the start of the whole experiment
    smooth: bool = True             # A flag to turn on or off Savitzky-Golay smoothing of the neutron data
    savgol_window: int = 21                 # The Savitzky-Golay smoothing window size
    savgol_polyorder: int = 3               # The Savitzky-Golay smoothing polynomial order
    crop_time: bool = False                 # A flag to determine whether the entire experiment should be plotted (False), or a specific time period (True)
    start_time: Optional[str] = None        # The start date and time for a cropped time plot as a string in format "YYYY-MM-DD HH:MM:SS"
    end_time: Optional[str] = None          # The end date and time for a cropped time plot as a string in format "YYYY-MM-DD HH:MM:SS"
    tzero: bool = True                      # Setting this to True will set the first time point in the plot to be 0
    normalise_tzero: bool = True            # A flag to determine if you want to renormalise the first point in the neutron data to 1 for this cropped time period
    crop_limits: tuple[tuple[int, int], tuple[int, int]] = ((750, 1280), (1400, 550))       # A 2-tuple of 2-tuples to determine the crop area for a cropped image ((x-pixel top left, y-pixel top left),(width, height)). None = don't crop
    timestamp_colour: str = "white"                     # What colour should the timestamp be
    timestamp_loc: tuple[float, float] = (0.1, 0.92)    # Graph fractions x, y to locate the timestamp
    timestamp_font_size: int = 32                       # Font size for the timestamp
    axes_label_font_size: int = 18                      # Font size for the colourmap axis label
    tick_font_size: int = 12                            # Font size for the colourmap tick labels
    figure_size: tuple[int, int] = (8, 8)               # Colourmap figure size
    scale_bar: Optional[tuple[float, int]] = None               # Scalebar parameters as a tuple in the format (pixel scale in mm, scale bar size in mm), None = don't plot a scalebar
    scale_bar_pos: tuple[str, str] =  ("lower left", "white")   # Scalebar position and colour, see https://pypi.org/project/matplotlib-scalebar/
    bl: bool = False                        # A flag to determine whether to perform Beer-Lambert conversion of the neutron y-axis units
    bl_axis_title: str = "H$_2$ conc. (mmol cm$^{-3}$)"     # A title to give for the neutron y-axis if Beer-Lambert has been used
    samp_thickness: Optional[float] = None  # Neutron beam path length in cm for Beer-Lambert analysis
    total_xsection: Optional[float] = None  # Total neutron cross section to use for Beer-Lambert analysis, in barns
    
@dataclass(kw_only=True)
class DiffImg():
    run: int                # The run this colour map figure relates to, as an index from the run_data list
    output_subdir: str      # The subdirectory to save the outputs, under exp_outputs_path
    fig_name: str           # The name for the figure, used as filenames
    save_format: str = ".png"   # The file format to use for saved images, usually ".png" or ".tif"
    ref_image: str                  # Full path to the reference image filename to use under the colourmap
    conv_mins: bool = True          # A flag to determine if the timestamp should be converted to mins or remain in seconds
    frame_select: Optional[tuple[int, int]] = None      # A 2-tuple denoting the first and last frames to include from the run
    crop_limits: Optional[tuple[tuple[int, int], tuple[int, int]]] = None       # A 2-tuple of 2-tuples to determine the crop area for a cropped image ((x-pixel top left, y-pixel top left),(width, height)). None = don't crop
    timestamp_colour: str = "white"                     # What colour should the timestamp be
    timestamp_loc: tuple[float, float] = (0.1, 0.92)    # Graph fractions x, y to locate the timestamp
    timestamp_font_size: int = 32                       # Font size for the timestamp
    figure_size: tuple[int, int] = (8, 8)               # Colourmap figure size
    scale_bar: Optional[tuple[float, int]] = None               # Scalebar parameters as a tuple in the format (pixel scale in mm, scale bar size in mm), None = don't plot a scalebar
    scale_bar_pos: tuple[str, str] =  ("lower right", "black")  # Scalebar position and colour, see https://pypi.org/project/matplotlib-scalebar/
    noise_control: bool = False         # A flag to determine whether to apply a bilateral noise reduction filter
    stretch_pct: tuple[float, float] = field(default_factory=lambda: [2, 95])   # A 2-tuple denoting the lower and upper percentiles to use for a histogram stretch of the resulting images
    

