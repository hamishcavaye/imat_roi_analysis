"""
A configuration script to provide IMAT experiment variables to use when 
processing image data and plotting resulting figures for reports and
publication.

This script defines the dataclasses for importing and use in an experiment's 
configuration file.

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

from dataclasses import dataclass, field
from typing import Optional
from matplotlib.colors import Colormap
from matplotlib import colormaps

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

DEFAULT_LABELS = [["ms", "a", (0.95, 0.025)],
                  ["n", "b", (0.95, 0.025)],
                  ["inset", "c", (0.05, 0.05)],
                  ]

# Define the dataclasses to use
# One for the overall experiment config and data directories
# One for the individual runs
# One for plotting figures
@dataclass(kw_only=True)
class ExperimentConfig:
    experiment_name: str
    
    raw_data_path: str
    exp_outputs_path: str
    
    dark_paths: list
    flat_paths: list
    dark_flat_output_path: str        # subdirectory under exp_outputs_path
    
    csv_path: Optional[str] = None
    
    use_temp_data: bool = False
    temp_data_path: Optional[str] = None
    temp_data_output_path: Optional[str] = None
    temp_data_output_file: Optional[str] = None
    
    use_mass_spec_data: bool = False
    mass_spec_data_path: Optional[str] = None
    mass_spec_data: Optional[list] = None
    mass_spec_data_output_path: Optional[str] = None
    mass_spec_data_output_file: Optional[str] = None
    mass_spec_time_offset: Optional[int] = None
    mass_spec_date_row: Optional[int] = None         # The 0 counted row number containing the start date and time
    mass_spec_start_row: Optional[int] = None        # The 0 counted row number containing the data headings "Time" etc.
    mass_spec_fieldnames: Optional[list] = None
    
    remove_outliers: bool = False
    bright_threshold: Optional[int] = None
    dark_threshold: Optional[int] = None
    outlier_radius: int = 3
    
    roi_grid: bool = False
    # Provide the size and shape of the area for the grid in the format
    # (top left x pixel, top left y pixel, width, height)
    grid_position: Optional[tuple[int, int, int, int]] = None
    grid_rows: Optional[int] = None
    grid_cols: Optional[int] = None
    # If not using an ROI grid:
    # Give the pixel coordinates of the top-left pixel for each ROI of interest.
    # Multiple ROI groups can be given here (it is a list of lists). 
    # Format is "(top left x pixel, top left y pixel, roi width, roi height)"
    roi_groups: Optional[list[list[tuple[int, int, int, int]]]] = None
    calib_topleft: tuple    # e.g. an air region for ROI normalisation or an internal standard point. Format is "(top left x pixel, top left y pixel, roi width, roi height)"
    standard_rois: Optional[list[tuple[int, int, int, int]]] = None
    
    colours: list[str] = field(default_factory=lambda: DEFAULT_COLOURS.copy())
    
    save_full_frames: bool = False
    
    save_reference_frame: bool = True
    reference_frame_range: list = range(0,20)
    use_scale_bar: bool = False
    scale_bar: Optional[tuple[float, int]] = None
    scale_bar_pos: tuple = ("lower right", "black")
    
@dataclass(kw_only=True)
class RunConfig():
    data_subdir: str
    csvlog: Optional[str] = None
    truncate_image_list: Optional[list[int]] = None
    truncate_log: Optional[list[int]] = None
    roi_list: int = 0
    output_name: str
    frame_skip: Optional[int] = None
    
@dataclass(kw_only=True)
class FigPars():
    output_subdir: str
    fig_name: str
    plot_mode: str = "neutron_only"
    conv_mins: bool = True
    normalise: bool = True
    smooth: bool = True
    savgol_window: int = 21
    savgol_polyorder: int = 3
    crop_time: bool = False
    start_time: Optional[str] = None
    end_time: Optional[str] = None
    tzero: bool = True
    normalise_tzero: bool = True
    mask_ndata: bool = False
    mask_start: Optional[str] = None
    mask_end: Optional[str] = None
    mass_spec_pars: Optional[list[tuple[str, int, str, str]]] = None
    plot_temp: bool = False
    smooth_temp: bool = False
    plot_inset: bool = False
    inset_axes_loc: Optional[list[float, float, float, float]] = None
    inset_axes_xlim: Optional[list[int, int]] = None
    inset_axes_ylim: Optional[list[float, float]] = None
    custom_ms_ylimits: Optional[tuple[float, float]] = None
    custom_neutron_ylimits: Optional[tuple[float, float]] = None
    custom_temp_ylimits: Optional[tuple[int, int]] = None
    axes_label_font_size: int = 64
    tick_font_size:int  = 64
    figure_size: tuple = (40, 20)
    double_figure_size: tuple = (60, 40)
    mass_spec_label_size: int = 60
    plot_arrow: Optional[list[list[int, str, str, float, float]]] = None
    plot_vline: Optional[list[int]] = None
    subplot_labels: list[list[str, str, tuple[float, float]]] = field(default_factory=lambda: DEFAULT_LABELS.copy())
    subplot_label_size: int = 80
    plot_line_width: int = 8
    
@dataclass(kw_only=True)
class ColourMapFig():
    run: int
    output_subdir: str
    fig_name: str
    save_format: str = ".png"
    invert_scale: bool = True
    cbar_title: str = "Reduced Transmission (%)"
    cmap: Colormap = field(default_factory=lambda: colormaps["inferno"])
    cmap_norm: dict = field(default_factory=lambda: {"norm": "norm", "vmin": 0, "vmax": 0.12})   #"norm", "log", "twinslope"
    heatmap_alpha: float = 0.7     # Transparency to use on the heatmap overlay
    bar_shrink_factor: float = 0.9     # Adjusts the size of the heatmap colour legend/bar
    ref_image: str
    conv_mins: bool = True
    normalise: bool = True
    smooth: bool = True
    savgol_window: int = 21
    savgol_polyorder: int = 3
    crop_time: bool = False
    start_time: Optional[str] = None
    end_time: Optional[str] = None
    tzero: bool = True
    normalise_tzero: bool = True
    crop_image: bool = False
    crop_limits: tuple[tuple[int, int], tuple[int, int]] = ((750, 1280), (1400, 550))
    timestamp_colour: str = "white"
    timestamp_loc: tuple[float, float] = (0.1, 0.92)     # Graph fractions x, y to locate the timestamp
    timestamp_font_size: int = 32
    axes_label_font_size: int = 18
    tick_font_size: int = 12
    figure_size: tuple[int, int] = (8, 8)
    scale_bar: Optional[tuple[float, int]] = None      # (pixel scale in mm, scale bar size in mm)
    scale_bar_pos: tuple[str, str] =  ("lower left", "white")
    bl: bool = False    # Perform Beer-Lambert conversion?
    bl_axis_title: str = "H$_2$ conc. (mmol cm$^{-3}$)"
    samp_thickness: Optional[float] = None  # in cm
    total_xsection: Optional[float] = None  # in barns
    
@dataclass(kw_only=True)
class DiffImg():
    run: int
    output_subdir: str
    fig_name: str
    save_format: str = ".png"
    ref_image: str
    conv_mins: bool = True
    frame_select: Optional[tuple[int, int]] = None
    crop_limits: Optional[tuple[tuple[int, int], tuple[int, int]]] = None       # ((x, y), (w, h))
    timestamp_colour: str = "white"
    timestamp_loc: tuple[float, float] = (0.05, 0.95)     # Graph fractions x, y to locate the timestamp
    timestamp_font_size: int = 32
    figure_size: tuple[int, int] = (8, 8)
    scale_bar: Optional[tuple[float, int]] = None      # (pixel scale in mm, scale bar size in mm)
    scale_bar_pos: tuple[str, str] =  ("lower right", "black")
    noise_control: bool = False
    stretch_pct: tuple[float, float] = field(default_factory=lambda: [2, 95])