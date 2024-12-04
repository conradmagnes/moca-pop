# `MoCap-POPY` Changelog

This changelog is a record of all notable changes made to the `MoCap-POPY`.

Entries should be prepended to the list below before merging on to `main`.
Entries should have be named with the date of the merge in the format `YYYY-MM-DD`.
Each entry has to feature the following sections: **Added**, **Changed**,
**Removed**, and **Fixed**.
If no item is present for a given section, have it say None.

## 2024-12-04
### Added
- `mocap_popy/scripts/swap_rb_markers` - main script for swapping and removing marker labels from a subject with rigid bodies based on distance to a reference rigid body.
- in `mocap_popy/utils`:
  - `argparse_utils.py` - helper functions for parsing command line arguments (e.g. frame range, directories).
  - `vicon_utils.py` - helper functions for working with online Vicon data (e.g. get marker trajectories, frame numbers).

### Changed
- `unassign_rb_markers.py` updates:
  - updated docstrings
  - moved file writing before inspect and removals
  - moved offline/online args and frame range validation to separate module (`utils/argparse_utils.py`).
  - moved 'get_next_filename' function to `config/directory.py`.
- `interactive_score_analyzer` updates:
  - ported 'run_isa_subprocess' function from `unassign_rb_markers.py` to `app.py`
  - moved best_fit transform functions from `helpers.py` to `models/rigid_body.py`.

### Removed
None

## 2024-12-03
### Added
- `mocap_popy/scripts/nushu_pipeline_runner` - script for running the Nushu pipeline on a given dataset.
  - `nushu_pipeline_runner.py` - main script for running the pipeline
  - `batch_runner.py` - runs the pipeline on multiple datasets
- in `mocap_popy/utils`:
  - `quality_check.py` - script for checking the quality of Vicon data (e.g. marker gaps, percentage of labeled data)

### Changed
- minor docstring updates and bug fixes to Interactive Score Analyzer and Unassign Rigid Body Marker scripts.
- `unassign_rb_markers.py` updates:
  - `--max_markers_to_remove` flag for specifying the maximum number of markers to remove in a single frame.
- added functions to `MarkerTrajectory` model for calculating the percentage of labeled data and the number of gaps in the data.

### Removed
None

## 2024-12-01
### Added
- `mocap_popy/aux_scripts/interactive_score_analyzer` - launches a web-based (dash) app for visualizing and
developing scoring strategies for rigid bodies. Includes:
  - `app.py` - main script for launching the app
  - `layout.py` - layout of the app
  - `helpers.py` - helper functions for the app
  - `constants.py` - constants used in the app

### Changed
- Added inspect flag to `unassign_rb_markers.py` to launch interactive score analyzer after scoring

### Removed
None

## 2024-11-20
### Added
- `mocap_popy/utils/hmi.py` - helpers for interacting with command line
- `mocap_popy/scripts/unassign_rb_markers/scoring`
  - `saved_parameters/` - folder containing saved scoring parameters
  - `scoringParameters.py` - model for scoring params (pydantic)
      - includes aggregation parameters and score thresholds (including at node level)

### Changed
- Renamed and update `rigid_body_scorer.py` to `scorer.py` and moved to `mocap_popy/scripts/unassign_rb_markers/scoring`
  - Fixed duplicate scoring issue during sort task in `scorer.py` (include/exclude arg)
  - deprecated component-level thresholds before aggregation
  - fixed generation of prior residual histories in `scorer.py`
- Updated `unassign_rb_markers.py` to use `scorer.py` and `scoringParameters.py`

### Removed
None

## 2024-11-19
### Added
- `mocap_popy/utils/dist_utils.py` - helpers for running scripts from different computers

### Changed
- Renamed `aux` to `aux_scripts` to avoid Windows reserved word
- Use relative paths in `generate_template_file_mapper.py`
- Updated 'online' implementation of `unassign_rb_markers.py` (tested with ViconNexus)
  - Added flags for opening new console (helpful when running script from within Nexus)


### Removed
None

## 2024-11-18
### Added
- `example_datasets/shoe_stepping` - folder containing example data recorded during Conrad's MT
- Under `mocap_popy`
  - `aux/generate_template_file_mapper.py` - script for generating a json file that maps model names to their respective template file locations
  - `config/` - folder containing common configuration files
    - `directory.py` - defines paths to common directories
    - `logger.py` - module-wide logger configuration
    - `regex.py` - common regex patterns for parsing data
  - `models/` - folder containing custom models for use in the pipeline
    - `rigid_body.py` - contains objects defining a rigid body (Nodes, Segments, Joints, RigidBody)
    - `marker_trajectory.py` - interface classes defining a marker trajectory (used both online and offline)
  - `scripts/`
    - `unassign_rb_markers.py` - script for unassigning marker labels from a subject with rigid bodies based on segment and joint residuals (lengths and angles compared to calibrated or prior bodies)
  - `templates/` - folder containing model templates for writing / reading model objects from files
    - `json/` - folder containing for JSON templates relating to the models
    - `baseTemplate.py`
    - `rigidBodyTemplate.py`
  - `utils` - folder containing several files with utility functions
    - `c3d_parser.py` - functions for parsing C3D files
    - `json_utils.py` - functions for reading and writing JSON files
    - `model_template_loader.py` - functions for loading model templates
    - `plot_utils.py` - functions for plotting data
    - `rigid_body_loader.py` - functions for loading rigid body objects
    - `rigid_body_scorer.py` - functions for scoring rigid body objects based on residuals
    - `string_utils.py` - functions for string manipulation
    - `vsk_parser.py` - functions for parsing VSK files

### Changed
- `aux` renamed to `aux_scripts` to avoid Windows reserved word
- `pyproject.toml` updated to include new dependencies
- `README.md` updated

### Removed
- `tests` from `mocap_popy` directory (in favor of including `tests` dirs closer to the scripts they test)

## 2024-11-13
### Added
Changes for the initial repo setup were comitted directly to the main branch. 
The following directories were added:
- logs/
- mocap_popy/
  - aux/
  - scripts/
  - test/

The following files were added:
- .gitignore
- README.md
- CHANGELOG.md
- pyproject.toml

### Changed
None

### Removed
None