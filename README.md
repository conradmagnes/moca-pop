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

## Repo Structure
<pre>
.
|-- example_datasets - Example datasets
|-- logs - Logging
`-- mocap_popy
    |-- aux - Auxilary
    |-- config - Configuration
    |-- scripts
    |   |-- unlabel_using_custom_skeleton.py
    `-- tests - Testing
</pre>

## Usage


