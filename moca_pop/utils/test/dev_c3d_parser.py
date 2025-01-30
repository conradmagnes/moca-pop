# %%
import os
import unittest

import matplotlib.pyplot as plt
import numpy as np

import moca_pop.utils.c3d_parser as c3d_parser
import moca_pop.utils.plot_utils as plotUtils

THIS_PATH = os.path.abspath(os.path.dirname(__file__))
TEST_DATA_PATH = os.path.join(THIS_PATH, "data", "trial01.c3d")
TEST_PLT_DIR = os.path.join(THIS_PATH, "plts")

EXPECTED_POINT_LABELS = [
    "Right_HH",
    "Right_HB",
    "Right_ML",
    "Right_TL",
    "Right_TF",
    "Right_TH",
    "Left_HH",
    "Left_HB",
    "Left_HL",
    "Left_ML",
    "Left_TL",
    "Left_TF",
    "Left_TH",
    "Right_HL",
    "Right_TM",
    "Left_TM",
    "Left_HM",
    "Left_MM",
    "Right_MM",
    "Right_HM",
    "*20",
]

PLOT_MARKER = "Right_ML"


reader = c3d_parser.get_reader(TEST_DATA_PATH)
trajectories = c3d_parser.get_marker_trajectories(
    reader, marker_labels=EXPECTED_POINT_LABELS[:-1], unit_scale=0.001
)
positions, residuals = c3d_parser.get_marker_positions(
    reader, marker_labels=EXPECTED_POINT_LABELS[:-1], unit_scale=0.001
)
frames = c3d_parser.get_frames(reader)

fig, ax = plt.subplots(1, 2, figsize=(10, 4), sharex=True, sharey=True)

ax[0].plot(frames, trajectories[PLOT_MARKER].x, label="x")
ax[0].plot(frames, trajectories[PLOT_MARKER].y, label="y")
ax[0].plot(frames, trajectories[PLOT_MARKER].z, label="z")


# %%
non_existent = np.array(frames, trajectories[PLOT_MARKER].exists) == False
exists_l, exists_g = plotUtils.get_binary_signal_edges(non_existent)
plotUtils.draw_shaded_regions(
    ax[0], exists_l, exists_g, color="gray", alpha=0.5, label="non-existent"
)

pos_arr = np.array(positions[PLOT_MARKER])
ax[1].plot(pos_arr)

res_binarr = np.array(residuals[PLOT_MARKER]) != 0
res_l, res_g = plotUtils.get_binary_signal_edges(res_binarr)
plotUtils.draw_shaded_regions(
    ax[1], res_l, res_g, color="gray", alpha=0.5, label="residual != 0"
)

ax[0].set_title(f"{PLOT_MARKER} Trajectories")
ax[1].set_title(f"{PLOT_MARKER} Positions/Residuals")

for i in range(2):
    ax[i].set_xlabel("Frame")
    ax[i].set_ylabel("Position (m)")
    ax[i].legend()
