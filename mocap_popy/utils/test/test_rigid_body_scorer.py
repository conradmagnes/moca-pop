# %%

import logging
import copy
import os
import logging
import numpy as np
import matplotlib.pyplot as plt
import time

import mocap_popy.config.directory as directory
from mocap_popy.models.rigid_body import RigidBody, Node
from mocap_popy.models.marker_trajectory import MarkerTrajectory
from mocap_popy.utils import rigid_body_loader, rigid_body_scorer, c3d_parser

# %%
DATASET_DIR = os.path.join(directory.DATASET_DIR, "shoe_stepping")
TRIAL_FN = "trial01.c3d"
SUBJECT_NAME = "subject"

LOGGER = logging.getLogger("TestRigidBodyScorer")


c3d_fp = os.path.join(DATASET_DIR, TRIAL_FN)
vsk_fp = os.path.join(DATASET_DIR, f"{SUBJECT_NAME}.vsk")

calibrated_rigid_bodies = rigid_body_loader.get_rigid_bodies_from_vsk(vsk_fp)

# %%
c3d_reader = c3d_parser.get_reader(c3d_fp)
marker_trajectories = c3d_parser.get_marker_trajectories(c3d_reader)
frames = c3d_parser.get_frames(c3d_reader)

# %%

scoring = {
    "segment": {
        "weight": 1.0,
        "tolerance": 0.1,
    },
    "joint": {
        "weight": 1 / 360,
        "tolerance": 0.1,
    },
}

segments_only = True

# %%
start = time.time()

residual_histories = rigid_body_scorer.generate_residual_histories(
    calibrated_rigid_bodies,
    marker_trajectories,
    "calib",
    segments_only,
    agg_kwargs=None,
)

print(f"Time: {time.time() - start}")


# %%
start = time.time()

residual_histories = rigid_body_scorer.generate_residual_histories(
    calibrated_rigid_bodies,
    marker_trajectories,
    "calib",
    False,
    agg_kwargs=None,
)

print(f"Time: {time.time() - start}")

# %%
segments_only = True
nrow = 1 if segments_only else 2
ncol = len(calibrated_rigid_bodies)
fig, ax = plt.subplots(nrow, ncol, figsize=(5 * ncol, 4 * nrow), sharey="row")

seg_axs = ax if nrow == 1 else ax[0, :]
joint_axs = [] if segments_only else ax[1, :]

for i, (body_name, body) in enumerate(calibrated_rigid_bodies.items()):
    rf_seg = np.array(residual_histories[body_name]["segment"])
    seg_axs[i].plot(rf_seg)
    seg_axs[i].legend(calibrated_rigid_bodies[body_name].get_markers())
    seg_axs[i].set_title(body_name)
    seg_axs[i].set_ylabel("Segment Residual")

    if len(joint_axs) > 0:
        rf_joint = np.array(residual_histories[body_name]["joint"])
        joint_axs[i].plot(rf_joint)
        joint_axs[i].set_ylabel("Joint Residual")


# %% First Pass = Calibration
start = time.time()

segments_only = True
residual_type = "calib"
score_component = "segment" if segments_only else "node"
score_component = "node"

score_histories = {}
removal_binaries = {}
for rb_name, rb in calibrated_rigid_bodies.items():
    prior_rigid_body = None
    rb_trajs: dict[str, MarkerTrajectory] = {
        m: traj for m, traj in marker_trajectories.items() if m in rb.get_markers()
    }

    removal_binaries[rb_name] = []
    score_histories[rb_name] = []

    for frame in range(0, len(frames)):
        nodes = [traj.generate_node(m, frame) for m, traj in rb_trajs.items()]
        if residual_type == "calib" or prior_rigid_body is None:
            current_rigid_body = copy.deepcopy(rb)
        else:
            current_rigid_body = copy.deepcopy(prior_rigid_body)

        current_rigid_body.update_node_positions(
            nodes, recompute_lengths=True, recompute_angles=(not segments_only)
        )
        current_rigid_body.compute_segment_residuals(rb)
        if not segments_only:
            current_rigid_body.compute_joint_residuals(rb)

        node_scores = rigid_body_scorer.score_rigid_body_components(
            current_rigid_body,
            score_component,
            residual_type,
            return_composite_score=True,
            ignore_residual_tolerances=False,
        )
        worst_nodes = rigid_body_scorer.sort_marker_scores(
            node_scores, threshold=0, max_markers=1
        )
        max_iter = 3
        i = 0

        removals, scores = {}, {}
        for m in current_rigid_body.get_markers():
            removals[m] = False
            scores[m] = 0

        while worst_nodes and i < max_iter:
            node, score = worst_nodes[0]
            removals[node] = True
            scores[node] = score if isinstance(score, (float, int)) else score[0]

            current_rigid_body.remove_node(node)
            node_scores = rigid_body_scorer.score_rigid_body_components(
                current_rigid_body,
                score_component,
                residual_type,
                return_composite_score=True,
                ignore_residual_tolerances=False,
            )
            worst_nodes = rigid_body_scorer.sort_marker_scores(
                node_scores, threshold=0, max_markers=1
            )
            i += 1

        prior_rigid_body = current_rigid_body

        removal_binaries[rb_name].append(list(removals.values()))
        score_histories[rb_name].append(list(scores.values()))

print(f"Time: {time.time() - start}")
# %%
from mocap_popy.utils import plot_utils

ncols = len(removal_binaries)
nrows = len(calibrated_rigid_bodies[rb_name].get_markers())

fig, ax = plt.subplots(
    nrows, ncols, figsize=(5 * ncols, max(1, nrows)), sharex=True, sharey=True
)

for i, (rb_name, rb) in enumerate(calibrated_rigid_bodies.items()):
    binaries = np.array(removal_binaries[rb_name])
    for j, marker in enumerate(rb.get_markers()):
        arr = binaries[:, j]
        l, g = plot_utils.get_binary_signal_edges(arr)
        plot_utils.draw_shaded_regions(ax[j, i], l, g, color="blue", alpha=0.8)
        ax[j, i].set_title(marker)

        ax[j, i].set_yticks([])

fig.tight_layout()


# %%
binaries = np.array(removal_binaries["Right_Foot"])
arr = binaries[:, 0]
l, g = plot_utils.get_binary_signal_edges(arr)

# %%
# %%

fig, ax = plt.subplots(2, 1, figsize=(5, 8))

body_name = "Left_Foot"
res_type = "calib"
rf_seg = np.array(residual_histories[body_name]["segment"])
# rf_joint = np.array(residual_histories[body_name]["joint"])

ax[0].plot(rf_seg)
# ax[1].plot(rf_joint)
ax[0].legend(calibrated_rigid_bodies[body_name].get_markers())

# %%
fig, ax = plt.subplots(1, 1, figsize=(5, 5))

rf_seg_calib = np.array(residual_histories[body_name]["segment"]["calib"])
rf_seg_prior = np.array(residual_histories[body_name]["segment"]["prior"])

ax.plot(rf_seg_calib[:, -3:])
# ax.plot(rf_seg_prior[:, -3:])
# %%

calibrated_rigid_bodies["Right_Foot"].nodes
# %%
