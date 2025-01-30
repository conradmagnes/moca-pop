import logging

import os
import logging
import numpy as np
import matplotlib.pyplot as plt

import moca_pop.config.directory as directory
import moca_pop.utils.plot_utils as plot_utils
import moca_pop.utils.rigid_body_loader as rigid_body_loader
import moca_pop.utils.vsk_parser as vsk_parser

## Uncomment for testing shoe stepping dataset
DATASET_NAME = "shoe_stepping"
TRIAL_FN = "trial01.c3d"
SUBJECT_NAME = "subject"
RB_NAME = "Foot"

# DATASET_NAME = "foot_drop"
# TRIAL_FN = "tpose.c3d"
# SUBJECT_NAME = "subject"
# RB_NAME = "fai_wholefoot"
# RB_NAME = "rearfoot"
IGNORE_SYMMETRY = False
CUSTOM = False

DATASET_DIR = os.path.join(directory.DATASET_DIR, DATASET_NAME)
PLOT_DIR = os.path.join(directory.UTILS_DIR, "test", "plts")


LOGGER = logging.getLogger("TestRigidBodyLoader")

vsk_fp = os.path.join(DATASET_DIR, f"{SUBJECT_NAME}.vsk")

model, bodies, parameters, marker_names, sticks = vsk_parser.parse_vsk(vsk_fp)

if CUSTOM:
    rb_names = (
        [RB_NAME + f"_{side}" for side in ["L", "R"]]
        if not IGNORE_SYMMETRY
        else [RB_NAME]
    )
    rigid_bodies = rigid_body_loader.get_custom_rigid_bodies_from_vsk(
        vsk_fp, rb_names, IGNORE_SYMMETRY
    )
else:
    rigid_bodies = rigid_body_loader.get_rigid_bodies_from_vsk(
        vsk_fp, ignore_symmetry=IGNORE_SYMMETRY
    )
rigid_bodies = {k: v for k, v in rigid_bodies.items() if RB_NAME in k}

plane = "xy"
colors = ["b", "r"]

fig = plt.figure(figsize=(8, 4))
rigid_body_loader.plot_rigid_bodies_on_figure(fig, list(rigid_bodies.values()), plane)
fig.tight_layout()

suff = "sym" if not IGNORE_SYMMETRY else "asym"
fig_path = os.path.join(
    PLOT_DIR, f"test_rigid_bodies_{DATASET_NAME}_{RB_NAME}_{suff}.png"
)
fig.savefig(fig_path)
