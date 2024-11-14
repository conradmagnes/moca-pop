# %%
import logging

# %%


import os
import logging
import numpy as np
import matplotlib.pyplot as plt

import mocap_popy.config.directory as directory
import mocap_popy.config.logger as logger
import mocap_popy.utils.vsk_parser as vsk_parser
import mocap_popy.utils.c3d_parser as c3d_parser
import mocap_popy.models.rigid_body as rb


LOGGER = logging.getLogger(__name__)

## input args
SUBJECT_NAME = "subject"  # mandatory, default New Subject (e.g. unnamed)
RB_TEMPLATE_NAMES = "foot"  # comma separated list of body templates
OFFLINE = True  # default False, flag

## offline args
PROJECR_DIR = os.path.join(directory.DATASET_DIR, "shoe_stepping")

##
path_to_c3d = os.path.join(PROJECR_DIR, "trial01.c3d")
path_to_vsk = os.path.join(PROJECR_DIR, f"{SUBJECT_NAME}.vsk")


if not os.path.exists(path_to_c3d):
    LOGGER.error(f"Path to C3D file does not exist: {path_to_c3d}")
    raise FileNotFoundError

if not os.path.exists(path_to_vsk):
    LOGGER.error(f"Path to VSK file does not exist: {path_to_vsk}")
    raise FileNotFoundError

rb_template_list = RB_TEMPLATE_NAMES.split(",")
rb_templates = {}
for name in rb_template_list:
    rb_template = rb.RigidBodyTemplate(name)
    rb_templates[name] = rb_template

# %%
# if offline

model, bodies, parameters, markers, segments = vsk_parser.parse_vsk(path_to_vsk)


# %%
raw_marker_pos = vsk_parser.parse_marker_positions_by_segment(parameters)
marker_pos_calib = vsk_parser.format_marker_positions(raw_marker_pos, markers)


# %%
marker_trajectories = c3d_parser.get_marker_trajectories(path_to_c3d, path_to_vsk)
