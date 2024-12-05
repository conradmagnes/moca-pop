# %%

import logging
import copy
import os
import logging
import numpy as np
import matplotlib.pyplot as plt
import time

import moca_pop.config.directory as directory
from moca_pop.models.rigid_body import RigidBody, Node
from moca_pop.models.marker_trajectory import MarkerTrajectory
from moca_pop.scripts.unassign_rb_markers.scoring import scorer
from moca_pop.utils import rigid_body_loader, c3d_parser
from moca_pop.utils import plot_utils

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

valid_kwargs = scorer.validate_segment_and_joint_kwargs(
    {"test": False}, {"test": True, "test2": False}
)
valid_kwargs

valid_kwargs = scorer.validate_segment_and_joint_kwargs(
    {"segment": {"test": False}}, {"test": True, "test2": False}
)
valid_kwargs
# %%

segments_only = True

# %%
start = time.time()

residual_histories = scorer.generate_residual_histories(
    calibrated_rigid_bodies,
    marker_trajectories,
    "calib",
    segments_only,
    agg_kwargs=None,
)

print(f"Time: {time.time() - start}")


# %%
start = time.time()

residual_histories = scorer.generate_residual_histories(
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

        node_scores = scorer.score_rigid_body_components(
            current_rigid_body,
            score_component,
            residual_type,
            return_composite_score=True,
            ignore_residual_tolerances=False,
        )
        worst_nodes = scorer.sort_marker_scores(
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
            node_scores = scorer.score_rigid_body_components(
                current_rigid_body,
                score_component,
                residual_type,
                return_composite_score=True,
                ignore_residual_tolerances=False,
            )
            worst_nodes = scorer.sort_marker_scores(
                node_scores, threshold=0, max_markers=1
            )
            i += 1

        prior_rigid_body = current_rigid_body

        removal_binaries[rb_name].append(list(removals.values()))
        score_histories[rb_name].append(list(scores.values()))

print(f"Time: {time.time() - start}")
# %%


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
start = time.time()

segments_only = True
residual_type = "prior"
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

    prior_rb = copy.deepcopy(rb)
    seg_tols = {
        seg: tol * 0.5 for seg, tol in prior_rb.get_segment_tolerances().items()
    }
    prior_rb.update_segment_tolerances(seg_tols)
    ref_lengths = {str(seg): seg.length for seg in rb.segments}

    for frame in range(0, len(frames)):
        nodes = [traj.generate_node(m, frame) for m, traj in rb_trajs.items()]
        if prior_rigid_body is None:
            current_rigid_body = copy.deepcopy(prior_rb)
        else:
            current_rigid_body = copy.deepcopy(prior_rigid_body)

        current_rigid_body.update_node_positions(
            nodes, recompute_lengths=True, recompute_angles=(not segments_only)
        )
        current_rigid_body.compute_segment_residuals(
            None, prior_rigid_body, ref_lengths
        )
        if not segments_only:
            current_rigid_body.compute_joint_residuals(None, prior_rigid_body)

        node_scores = scorer.score_rigid_body_components(
            current_rigid_body,
            score_component,
            residual_type,
            return_composite_score=True,
            ignore_residual_tolerances=False,
        )
        worst_nodes = scorer.sort_marker_scores(
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
            node_scores = scorer.score_rigid_body_components(
                current_rigid_body,
                score_component,
                residual_type,
                return_composite_score=True,
                ignore_residual_tolerances=False,
            )
            worst_nodes = scorer.sort_marker_scores(
                node_scores, threshold=0, max_markers=1
            )
            i += 1

        prior_rigid_body = current_rigid_body

        removal_binaries[rb_name].append(list(removals.values()))
        score_histories[rb_name].append(list(scores.values()))

print(f"Time: {time.time() - start}")


# %%

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
