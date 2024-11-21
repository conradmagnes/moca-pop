import logging

import os
import logging
import numpy as np
import matplotlib.pyplot as plt

import mocap_popy.config.directory as directory
import mocap_popy.utils.plot_utils as plot_utils
import mocap_popy.utils.rigid_body_loader as rigid_body_loader

DATASET_DIR = os.path.join(directory.DATASET_DIR, "shoe_stepping")
PLOT_DIR = os.path.join(directory.UTILS_DIR, "test", "plts")
TRIAL_FN = "trial01.c3d"
SUBJECT_NAME = "subject"

LOGGER = logging.getLogger("TestRigidBodyLoader")

vsk_fp = os.path.join(DATASET_DIR, f"{SUBJECT_NAME}.vsk")

rigid_bodies = rigid_body_loader.get_rigid_bodies_from_vsk(
    vsk_fp, ignore_marker_symmetry=True
)

plane = "xy"
colors = ["b", "r"]

fig = plt.figure(figsize=(8, 4))
rigid_body_loader.plot_rigid_bodies_on_figure(fig, list(rigid_bodies.values()), plane)
fig.tight_layout()

fig_path = os.path.join(PLOT_DIR, "test_rigid_bodies.png")
fig.savefig(fig_path)
