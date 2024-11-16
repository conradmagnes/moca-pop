# %%

import logging

import os
import logging
import numpy as np
import matplotlib.pyplot as plt


import mocap_popy.config.directory as directory
from mocap_popy.utils import rigid_body_loader, plot_utils

# %%
DATASET_DIR = os.path.join(directory.DATASET_DIR, "shoe_stepping")
TRIAL_FN = "trial01.c3d"
SUBJECT_NAME = "subject"

LOGGER = logging.getLogger("TestRigidBodyScorer")


c3d_fp = os.path.join(DATASET_DIR, TRIAL_FN)
vsk_fp = os.path.join(DATASET_DIR, f"{SUBJECT_NAME}.vsk")

rigid_bodies = rigid_body_loader.get_rigid_bodies_from_vsk(vsk_fp)
