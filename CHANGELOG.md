# `MoCap-POPY` Changelog

This changelog is a record of all notable changes made to the `MoCap-POPY`.

Entries should be prepended to the list below before merging on to `main`.
Entries should have be named with the date of the merge in the format `YYYY-MM-DD`.
Each entry has to feature the following sections: **Added**, **Changed**,
**Removed**, and **Fixed**.
If no item is present for a given section, have it say None.

## 2024-11-19
### Added
- `utils/dist_utils.py` - helpers for running scripts from different computers

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