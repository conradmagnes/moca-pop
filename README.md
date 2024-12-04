# MOCAP-POPY: MOtion CAPture Pipeline Operations (using) PYthon

## Description
This repository contains python scripts for processing MoCap data.

## Installation

This package was developed and tested using Python 3.11.9. It is recommended to use a virtual environment to manage dependencies.

To create a virtual environment, run the following commands

```bash
# create a virtual environment
# If installing the venv in the mocap_popy directory, name the environment '.venv' so it is excluded from git.
python -m venv <env_name> 

# activate the virtual environment
# for Windows
.\<env_name>\Scripts\activate
# for Unix or MacOS
source <env_name>/bin/activate
```

To install this package, clone from git and install using pip (shown below). Installing the package will also install its dependencies.

```bash
# clone the repository
git clone git@github.com:conradmagnes/mocap_popy.git

# navigate to the repository and install the package
cd mocap_popy
python -m pip install .

# to install in editable mode, add the following flag
python -m pip install -e .
```

### Additional Dependencies

This library also requires the Vicon Nexus API, if running scripts in 'online' mode. 
The Vicon Nexus API is automatically downloaded with newer versions (>=2.12) of the Vicon Nexus software. 
Locate the library in Program Files (either Program or Program Files (x86)) under Vicon/Nexus<VerionNumber>/SDK/<OS>/Python. 
If using Windows, you can install the package by running the included `.bat` file (i.e. `install_viconnexusapi.bat`).
Otherwise, navigate to the folder containing the package (i.e. viconnexusapi) and run `python -m pip install ./<package_name>`. 
If using a virual environment, make sure to activate it before installing the package.

### Vicon Nexus Configuration

Python scripts can be run in Vicon Nexus using the 'Run Python Operation' pipeline operation (under 'Data Processing').
Make sure the operation points to the correct python executable and the correct script file location. 
If using a virtual environment, add the location of the virtual environment to the 'Environment activation' field (available in Vicon Nexus 2.16).
Otherwise, make sure all dependencies are installed in the correct python environment.


## Repo Structure
<pre>
.
|-- example_datasets                : contains example datasets for demo purposes
|   `-- shoe_stepping   
|-- logs                            : contains log files generated by the scripts
`-- mocap_popy                      : main package
    |-- aux_scripts                     : auxiliary scripts
    |   `-- interactive_score_analyzer
    |-- config                          : package configuration
    |-- model_templates                 : templates for generating models
    |   `-- json
    |-- models                          : data models
    |-- scripts                         : main scripts
    |   |-- pipeline_runner
    |   `-- unassign_rb_makers          
    `-- utils                           : utility functions
    
</pre>

## Main Scripts

### 1. `pipeline_runner.py`

This script executes a series of pre-defined pipelines. 
This script must be run as a standalone script, and cannot be integrated into a Vicon Nexus pipeline (see Vicon SDK documentation for RunPipeline()).

To view usage information, run the following command:
```bash
python mocap_popy/scripts/pipeline_runner/pipeline_runner.py --help
```

A batch runner is also provided to execute pipeline runners for several trials in the same project directory. See `batch_runner.py` for more information.

### 2. `unassign_rb_makers.py`

This script unassigns maker labels from a subject in a Vicon Nexus trial using a RigidBody model. A RigidBody defines
the relationship between markers in a Vicon Skeleton (vsk) model through Segments (lines) and Joints (angles). 
The script calculates the difference in segments lengths and joint angles between a calibrated Rigid Body model and the 
Rigid Body at each frame in the trial. Alternatively, the script can compare the Rigid Body between successive frames.
The script scores each marker based on differences in segment lengths and joint angles, and unassigns markers with
scores above a set threshold. The script can be run 'online', where it connects to Vicon Nexus and processes the data in
the active trial, or 'offline', where it reads data from a C3D file.

This file can be integrated into a Vicon Nexus pipeline by running the script in 'online' mode.

To view usage information, run the following command:
```bash
python mocap_popy/scripts/unassign_rb_markers/unassign_rb_markers.py --help
```

## Auxiliary Scripts

### 1. `interactive_score_analyzer.py`

This script allows the user to interactively analyze scoring strategies used by the `unassign_rb_markers.py` script.
The script launches a web-based Dash app that allows the user to manipulate nodes in a RigidBody model and view the
effects on the scoring algorithm. The user can adjust scoring parameters and save the results to a file for later use.
This script can also be run directly from `unassign_rb_markers.py` by including the `--inspect` flag.

Note, this script does not require internet access to run, as it is a local web app. It can be run both in 'online' and 'offline' modes, which refer to the source of the data being analyzed (i.e. Vicon Nexus or C3D file), and not to internet availability.

To view usage information, run the following command:
```bash
python mocap_popy/aux_scripts/interactive_score_analyzer/interactive_score_analyzer.py --help
```

## Contributing

When contributing to this repository, please first create a new branch (`git checkout -b <branch_name>`) and then submit a pull request. Assign the pull request to the repository owner for review.