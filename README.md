
# <img src="./media/images/moca-pop_icon.png" alt="moca-pop_icon" height="80" align="left"><span><strong>MOCA POP</strong><br><small>MOtion CApture Python Operation Pipelines</small></span>

## Description

This repository contains python scripts for processing motion capture data. The scripts were designed and tested
to work with the Vicon Nexus software or files exported from the software (.VSK, .C3D). The core concepts of this repository, however, can be
extended to work with other motion capture systems.

## Contents
- [Motivating Examples](#motivating-examples)
- [Installation](#installation)
- [General Usage](#general-usage)
- [Core Concepts](#core-concepts)
- [Main Scripts](#main-scripts)
- [Auxiliary Scripts](#auxiliary-scripts)
- [Contributing](#contributing)
- [Repo Structure](#repo-structure)


## Motivating Examples

### Marker Label Jumping and Swapping

Incompete or erroneous marker reconstruction can impact the labeling accuracy of out-of-the-box labeling pipelines used in motion capture software, such as Vicon Nexus. A common result is that marker labels "jump" to inaccurate positions, or are swapped with other markers. This happens often when ghost markers or other artifacts are present in the reconstruction. These issues may persist even after calibrating a labeling skeleton.

The following demo shows the sequential execution of the `swap_rb_markers.py` and `unassign_rb_markers.py` scripts to help resolve labeling errors. The demo uses a recording of a subject stepping forward with the right foot. Ten markers were placed on each of the subject's shoes. Labeling for the right shoe is colored, with notable mislabeled markers highlighted in red, yellow, blue and light green. Other marker labels are colored dark green (right shoe) or dark gray (left shoe). Light gray markers indicate unlabeled markers.

The `swap_rb_markers.py` script is executed first to identify and correct mislabeled markers (e.g. from t = 20 - 27 s). The `unassign_rb_markers.py` script is then executed to unassign markers that do not match the calibrated RigidBody model (e.g. between t = 2 - 8 s). The same video is provided in .MP4 format in the `media` directory.

![demo](/media/videos/moca-pop_main_demo.gif)


## Installation

This package was developed and tested using Python 3.11.9. It is recommended to use a virtual environment to manage dependencies.

To create a virtual environment, run the following commands

```bash
# create a virtual environment
# If installing the venv in the moca_pop directory, name the environment '.venv' so it is gitignored.
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

## General Usage

The scripts in this repository can be run in two modes: 'online' and 'offline'.
- 'Online' mode refers to running the script in the Vicon Nexus software, where the script connects to the active trial in the software.
- 'Offline' mode refers to running the script outside of the Vicon Nexus software, where the script reads data from a C3D file.

The 'online' mode requires the Vicon Nexus SDK to be installed and accessible to the script. Some 'online' scripts can be integrated directly into the Vicon Nexus software as a pipeline operation.

All scripts have a `--help` flag that provides information on how to use the script.

## Core Concepts

### The RigidBody Model

The RigidBody model uses Nodes, Segments, and Joints to define the relationship between markers in a motion capture trial.
Nodes represent marker positions in 3D space, Segments represent lines connecting two Nodes, and Joints represent angles between two Segments.

The Vicon Nexus labeing skeleton (VSK) uses a similar structure, but at one level of abstraction higher. VSK segments include three or more
markers. VSK joints define a relationship between two segments using one marker on each segment.

In this repo, a RigidBody is most similar to a segment in the VSK. A Segment in the RigidBody model is most similar to a stick in the VSK, which
is only used for visualization purposes.

The purpose of the RigidBody model is to enforce stricter constraints on marker labeling. By assuming a collection of markers are placed on a
body that is *more or less* rigid, we can calibrate a RigidBody model and use it to identify mislabeled markers in a trial.

'Standard' rigid bodies use only nodes fully contained within a VSK segment. 
These can be read and calibrated directly from the VSK file. 

'Custom' rigid bodies are those which include markers from more than one VSK segment. These must be added explicitly when calling the main scripts below by using the `--custom` or `--custom_rbs` flags. Model templates for 'custom' rigid bodies can remain agnostic to left and right symmetry, but the user should then specify the side when adding the name to the script call (i.e. 'fai_wholefoot' -> 'fai_wholefoot_R'). Further, the user should not use the `--ignore_symmetry` flag. This does not apply if all symmetry labels are included in the model template. Finally, when using 'custom' rigid bodies, the user should specify a 'static' frame for calibration. This is a frame where all markers are present and in the expected position. This is necessary as joints between segments in the VSK do not enforce a fixed distance between markers, resulting in rigid bodies overlapping near the origin. This can be done by using the `--static_frame` flag.

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

### 1. `interactive_score_analyzer/app.py`

The script launches a web-based application ([Dash](https://dash.plotly.com/)) that allows the user to manipulate nodes in a RigidBody model and view the
effects on the scoring algorithm. The user can adjust scoring parameters and save the results to a file for later use. RigidBody scoring is used heavily in the `unassign_rb_markers.py` script.

This script can also be run directly from the `unassign_rb_markers.py` and `swap_rb_markers.py` scripts by including the `--inspect` flag.

The sciprt supports both 'online' and 'offline' modes.
Note, this script does not require internet connectivity to run, as it is a local web app (not to be confused with the 'online' and 'offline' monikers).

To view usage information, run the following command:
```bash
python moca_pop/aux_scripts/interactive_score_analyzer/app.py --help
```

### 2. `pipeline_runner.py`

This script executes a series of pre-defined Vicon Nexus pipelines. 
It must be run online and as a standalone script while the user has the Vicon Nexus software open. A trial does not need to be active in the software for the script to run (granted the paths to the desired project and trial are specified).
It cannot be integrated directly into the Vicon Nexus software (e.g. as a pipeline). See `RunPipeline()` in the Vicon SDK for more information.

The script reads a configuration file that defines the pipelines to be executed. The `Pipeline` and `PipelineSeries` classes (in `pipeline_runner/pipeline.py`) 
allow for the specification of conditions to check before running a pipeline, and for the specification of a series of operations to be executed in sequence.

A batch runner is also provided to execute pipeline runners for several trials in the same project directory.

To view usage information, check the following:
```bash
python moca_pop/aux_scripts/pipeline_runner/pipeline_runner.py --help
python moca_pop/aux_scripts/pipeline_runner/batch_runner.py run --help
```

### 3. generate_template_file_mapping.py

This script iterates through the template directory and generates a json file that maps model and template names 
to template file paths. This mapping file is used by several scripts in the package to locate the correct template file for a given model.

To view usage information, run the following command:
```bash
python moca_pop/scripts/generate_template_file_mapping.py --help
```

### 4. print_trial_names.py

This script prints, or optionally writes to file, the names of all trials in a specified project directory. 
This is useful for quickly identifying the names of trials in a project directory, or to create a ledger when
using the `batch_runner.py` script.

To view usage information, run the following command:
```bash
python moca_pop/scripts/print_trial_names.py --help
```

## Contributing

When contributing to this repository, please first create a new branch (`git checkout -b <branch_name>`) before making changes. To make changes to the `main` branch, submit a pull request and assign it to the repository owner for review.

## Repo Structure
<pre>
.
|-- example_datasets/                : contains example datasets for demo purposes
|   `-- shoe_stepping/   
|-- example_pipelines/               : contains example pipeline files
|   `-- ETH_NUSHU/
|-- logs/                            : contains log files generated by the scripts
|-- media/                           : contains media files (e.g. images, videos)
`-- moca_pop/                        : main package
    |-- aux_scripts/                     : auxiliary scripts
    |   |-- interactive_score_analyzer/
    |   |-- pipeline_runner/
    |   |-- generate_template_file_mapping.py
    |   `-- print_trial_names.py
    |-- config/                          : package configuration
    |-- model_templates/                 : templates for generating models
    |-- models/                          : data models
    |-- scripts/                         : main scripts
    |   |-- swap_rb_markers/
    |   `-- unassign_rb_makers/          
    `-- utils/                           : utility functions
    
</pre>
