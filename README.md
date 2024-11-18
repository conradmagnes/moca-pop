# MOCAP-POPY: MOtion CAPture Pipeline Operations (using) PYthon

## Description
This repository contains python scripts for processing MoCap data.

## Installation

Clone this repo and install using pip. 

```bash
git clone git@github.com:conradmagnes/mocap_popy.git

cd mocap_popy
python -m pip install .

# to install in editable mode, use the following command
python -m pip install -e .

# to install optional test dependencies, use the following command
python -m pip install .[test]
# in zsh, escape the brackets using
python -m pip install .\[test\]
```

This library also requires the Vicon Nexus API, if running scripts in 'online' mode. The Vicon Nexus API is automatically downloaded with newer versions of the Vicon Nexus software (located in Program Files (either Program or Program Files (x86)) -> Vicon -> NexusX.XX -> SDK -> (OS) -> Python). If using Windows, you can install the package by running the included `.bat` file. 
Otherwise, navigate to the folder containing the package (i.e. viconnexusapi) and run `pip install .`.

## Repo Structure
<pre>
.
|-- example_datasets
|   `-- shoe_stepping 
|-- logs - Logging
`-- mocap_popy
    |-- aux_scripts
    |-- config
    |-- models
    |-- scripts
    |   `-- unassign_rb_makers
    |-- templates
    |   `-- json
    `-- utils
    
</pre>

## Usage


