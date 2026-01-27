# Region of Interest (ROI) Analysis Workflow for Neutron Radiography on IMAT

This repository contains a modular Python workflow for processing, calibrating, and analysing neutron radiography data from IMAT experiments. It has been designed with the operando study of heterogeneous catalysts in mind, but may be useful for other radiography experiments where neutron transmission across a set of regions of interest (ROIs) is monitored.

The workflow performs:

- Dark/flat frame correction
- Parsing of data from a mass spectrometer
- Parsing of temperature logs
- Image calibration and ROI extraction
- Multi-run time-aligned figure plotting

The project has been designed to be run in a Conda environment (e.g. via Spyder) and uses a configuration-driven structure through Python dataclasses. These configs can be tailored for a specific experiment, including numerous "runs" and multiple output figures with different parameters.

## Repository Structure
```
imat_roi_analysis/
|
|-- scripts/
|   |-- 01_Dark_Flat_Processing.py
|   |-- 02a_MassSpec_Parsing.py
|   |-- 02b_Temp_Parsing.py
|   |-- 03_Image_Processing.py
|   |-- 04_Figure_Plotting.py
|
|-- config/
|   |-- class_defs.py
|   |-- example_cfg.py
|
|-- examples/
|   |-- example-ms.csv
|   |-- example-temp.csv
|
|-- requirements.txt
|-- environment.yml
|-- README.md
|-- LICENSE
```
## Running the Workflow

Each script is numbered in the order it should be executed:

- 01_Dark_Flat_Processing.py
    - Generates master_dark.tif and dark_subtracted_master_flat.tif
- 02a_MassSpec_Parsing.py
    - Produces a pickled mass-spec time series if desired
- 02b_Temp_Parsing.py
    - Produces a pickled temperature time series if desired
- 03_Image_Processing.py
    - Applies dark and flat calibration and ROI extraction
    - Saves per-run time-series data and optional full calibrated images
- 04_Figure_Plotting.py
    - Loads outputs from all previous steps
    - Produces publication quality figures

All configuration values are supplied through a number of Python dataclasses in config/example_cfg.py

## Example Data

Examples of the used inputs for mass spec and temperature logs can be found in /examples

## Example Outputs

![An example image](/assets/example_ref_image.png)  
![An example figure](/assets/example_figure.png)  

## License

See LICENSE. This code is shared under the GNU GENERAL PUBLIC LICENSE Version 3.

## Author

Hamish Cavaye  
ISIS Neutron and Muon Source  
Science and Technology Facilities Council

