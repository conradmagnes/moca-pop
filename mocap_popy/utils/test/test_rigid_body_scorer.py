# %%

import logging

import os
import logging
import numpy as np
import matplotlib.pyplot as plt

import mocap_popy.config.directory as directory
import mocap_popy.config.logger as logger
import mocap_popy.config.regex as regex
import mocap_popy.utils.vsk_parser as vsk_parser
import mocap_popy.utils.c3d_parser as c3d_parser
import mocap_popy.models.rigid_body as rb
import mocap_popy.utils.json_utils as json_utils

import mocap_popy.utils.model_template_loader as model_template_loader


# %%
DATASET_DIR = os.path.join(directory.DATASET_DIR, "shoe_stepping")
TRIAL_FN = "trial01.c3d"
SUBJECT_NAME = "subject"

LOGGER = logging.getLogger("TestRigidBodyScorer")


c3d_fp = os.path.join(DATASET_DIR, TRIAL_FN)
vsk_fp = os.path.join(DATASET_DIR, f"{SUBJECT_NAME}.vsk")

reader = c3d_parser.get_reader(c3d_fp)
marker_trajectories = c3d_parser.get_marker_trajectories(reader)

# %%

model, bodies, parameters, markers, segments = vsk_parser.parse_vsk(vsk_fp)

# %%
markers_positions = vsk_parser.parse_marker_positions_by_segment(parameters)
# %%

body_templates = {}
for body in bodies:
    asym_body_name = body
    try:
        side, asym_body_name = regex.parse_symmetrical_component(body)
    except ValueError:
        LOGGER.warning(f"Body {body} is not a symmetrical component.")
    body_template = model_template_loader.load_template_from_json(
        model, "rigid_body", asym_body_name
    )
    if body_template is None:
        LOGGER.error(f"Failed to load template for body {body}.")
        continue

    body_template.add_symmetry_prefix(side)
    body_templates[body] = body_template

# %%
body_templates["Right_Foot"]
# %%
