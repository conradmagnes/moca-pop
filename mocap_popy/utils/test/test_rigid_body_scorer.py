# %%

import logging

import os
import logging
import numpy as np
import matplotlib.pyplot as plt

import mocap_popy.config.directory as directory
import mocap_popy.config.logger as logger
import mocap_popy.utils.vsk_parser as vsk_parser
import mocap_popy.utils.c3d_parser as c3d_parser
import mocap_popy.models.rigid_body as rb


DATASET_DIR = os.path.join(directory.DATASET_DIR, "shoe_stepping")
TRIAL_FN = "trial01.c3d"
SUBJECT_NAME = "subject"

LOGGER = logging.getLogger(__name__)


c3d_fp = os.path.join(DATASET_DIR, TRIAL_FN)
vsk_fp = os.path.join(DATASET_DIR, f"{SUBJECT_NAME}.vsk")

reader = c3d_parser.get_reader(c3d_fp)
marker_trajectories = c3d_parser.get_marker_trajectories(reader)

# %%

model, bodies, parameters, markers, segments = vsk_parser.parse_vsk(vsk_fp)

# %%
vsk_parser.parse_marker_positions_by_segment(parameters)
# %%
