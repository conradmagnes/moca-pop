# MOCA-POP: MOtion CApture Pipeline Operations using Python

## Description

This repository contains python scripts for processing motion capture data. The scripts were designed and tested
to work with the Vicon Nexus software or files exported from the software (.VSK, .C3D). However, the repository can be
extended to work with other motion capture systems.

## Contents
- [Demos](#demos)
- [Installation](#installation)
- [Repo Structure](#repo-structure)
- [Main Scripts](#main-scripts)
- [Auxiliary Scripts](#auxiliary-scripts)
- [Contributing](#contributing)


## Demos

### Sequential Pipeline Operations: `swap_rb_markers.py` and `unassign_rb_markers.py` 

The following demo shows the sequential execution of the `swap_rb_markers.py` and `unassign_rb_markers.py` scripts. The demo uses a recording of a subject stepping forward with the right foot. Ten markers were placed on each of the subject's shoes. Labeling for the right shoe is colored, with notable mislabeled markers highlighted in red, yellow, blue and light green. Other marker labels are colored dark green (right shoe) or dark gray (left shoe). Light gray markers indicate unlabeled markers.

The `swap_rb_markers.py` script is executed first to identify and correct mislabeled markers (starting at t = 20 s). The `unassign_rb_markers.py` script is then executed to unassign markers that do not match the calibrated RigidBody model (e.g. between t = 2 - 8 s). An .MP4 file can be found in the `media` directory.

![demo](/media/moca-pop_main_demo.gif)


## Installation

This package was developed and tested using Python 3.11.9. It is recommended to use a virtual environment to manage dependencies.

To create a virtual environment, run the following commands

```bash
# create a virtual environment
# If installing the venv in the moca_pop directory, name the environment '.venv' so it is excluded from git.
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
git clone git@github.com:conradmagnes/moca_pop.git

# navigate to the repository and install the package
cd moca_pop
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
|-- media                           : contains media files (e.g. images, videos)
`-- moca_pop                        : main package
    |-- aux_scripts                     : auxiliary scripts
    |   |-- interactive_score_analyzer
    |   `-- pipeline_runner
    |-- config                          : package configuration
    |-- model_templates                 : templates for generating models
    |-- models                          : data models
    |-- scripts                         : main scripts
    |   |-- swap_rb_markers
    |   `-- unassign_rb_makers          
    `-- utils                           : utility functions
    
</pre>



## General Usage

The scripts in this repository can be run in two modes: 'online' and 'offline'.
- 'Online' mode refers to running the script in the Vicon Nexus software, where the script connects to the active trial in the software.
- 'Offline' mode refers to running the script outside of the Vicon Nexus software, where the script reads data from a C3D file.

The 'online' mode requires the Vicon Nexus SDK to be installed and accessible to the script. Some 'online' scripts can be integrated directly into the Vicon Nexus software as a pipeline operation.

## Main Scripts

### 1. `swap_rb_markers.py`

This script scans a trial after labeling to identify markers that have been mislabeled.
This often occurs when the labeling algorithm assigns a marker to artifacts or ghost markers, causing downstream labeling issues.
The script identifies markers that are likely mislabeled by comparing marker positions to the expected positions based on a calibrated RigidBody model.
The script then swaps the mislabeled markers with the nearest markers in the model or unassigns them if no suitable marker is found. For more information, consult the script documentation.

The script supports both 'online' and 'offline' modes, as well as direct integration into the Vicon Nexus software as a pipeline operation.

To view usage information, run the following command:
```bash
python moca_pop/scripts/swap_rb_markers/swap_rb_markers.py --help
```

### 2. `unassign_rb_makers.py`

This script unassigns maker labels from a subject in a Vicon Nexus trial using a RigidBody model. A RigidBody defines
the relationship between markers in a Vicon Skeleton (vsk) model through Segments (lines) and Joints (angles). 
The script calculates the difference in segments lengths and joint angles between a calibrated Rigid Body model and the 
Rigid Body at each frame in the trial. Alternatively, the script can compare the Rigid Body between successive frames.
The script scores each marker based on differences in segment lengths and joint angles, and unassigns markers with
scores above a set threshold. 
The script supports both 'online' and 'offline' modes, as well as direct integration into the Vicon Nexus software as a pipeline operation.

To view usage information, run the following command:
```bash
python moca_pop/scripts/unassign_rb_markers/unassign_rb_markers.py --help
```

## Auxiliary Scripts

### 1. `interactive_score_analyzer.py`

The script launches a web-based application ([Dash](https://dash.plotly.com/)) that allows the user to manipulate nodes in a RigidBody model and view the
effects on the scoring algorithm. The user can adjust scoring parameters and save the results to a file for later use. RigidBody scoring is used heavily in the `unassign_rb_markers.py` script.

This script can also be run directly from the `unassign_rb_markers.py` and `swap_rb_markers.py` scripts by including the `--inspect` flag.

The sciprt supports both 'online' and 'offline' modes.
Note, this script does not require internet connectivity to run, as it is a local web app (not to be confused with the 'online' and 'offline' monikers).

To view usage information, run the following command:
```bash
python moca_pop/aux_scripts/interactive_score_analyzer/interactive_score_analyzer.py --help
```

### 1. `pipeline_runner.py`

This script executes a series of pre-defined Vicon Nexus pipelines. 
It must be run online and as a standalone script while the user has the Vicon Nexus software open. A trial does not need to be active in the software for the script to run (granted the paths to the desired project and trial are specified).
It cannot be integrated directly into the Vicon Nexus software (e.g. as a pipeline). See `RunPipeline()` in the Vicon SDK for more information.

The script reads a configuration file that defines the pipelines to be executed. The `Pipeline` and `PipelineSeries` classes (in `pipeline_runner/pipeline.py`) 
allow for the specification of conditions to check before running a pipeline, and for the specification of a series of operations to be executed in sequence.

A batch runner is also provided to execute pipeline runners for several trials in the same project directory.

To view usage information, check the following:
```bash
python moca_pop/scripts/pipeline_runner/pipeline_runner.py --help
python moca_pop/scripts/pipeline_runner/batch_runner.py run --help
```

## Contributing

When contributing to this repository, please first create a new branch (`git checkout -b <branch_name>`) and then submit a pull request.
Assign the pull request to the repository owner for review.