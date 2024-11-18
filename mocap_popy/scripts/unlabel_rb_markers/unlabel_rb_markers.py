"""!

    Unlabel Rigid Body Markers
    ===========================

    This script removes markers from rigid bodies based on residual segment and (optionally) joint scores.

    @author C. McCarthy
"""

import argparse
import logging
import copy
import os
import logging
import numpy as np
import matplotlib.pyplot as plt
import time
from typing import Literal, Union

import mocap_popy.config.directory as directory
from mocap_popy.models.rigid_body import RigidBody, Node
from mocap_popy.models.marker_trajectory import MarkerTrajectory
from mocap_popy.utils import rigid_body_loader, rigid_body_scorer, c3d_parser
from mocap_popy.utils import plot_utils


LOGGER = logging.getLogger("UnlabelRigidBodyMarkers")


def get_marker_removals_from_calibration_bodies(
    rigid_bodies: dict[str, RigidBody],
    marker_trajectories: dict[str, MarkerTrajectory],
    segments_only: bool,
    scoring_params: dict = None,
    agg_kwargs: dict = None,
    max_removals_per_frame: int = 3,
) -> tuple[dict[str, list], dict[str, list]]:
    """!Get marker removals from calibration bodies.

    Calculates the worst markers per frame and removes them from the rigid body.
    Returns a dictionary of binary arrays indicating which markers were removed per frame.
    Also returns a dictionary of scores for each marker per frame.

    @param rigid_bodies dict of RigidBody instances.
    @param marker_trajectories dict of MarkerTrajectory instances.
    @param segments_only If True, only segment residuals are considered.
    @param scoring_params dict of kwargs for scoring_rigid_body_components. (i.e. score_tolerance, score_weight)
    @param agg_kwargs dict of kwargs for aggregate_residuals. (i.e. average, ignore_tolerance)
    @param max_removals_per_frame Maximum number of markers to remove per frame.
    """

    residual_type = "calib"
    score_component = "segment" if segments_only else "node"

    scoring_params = rigid_body_scorer.validate_segment_and_joint_kwargs(
        scoring_params, {"score_tolerance": 0, "score_weight": 1}
    )

    agg_kwargs = rigid_body_scorer.validate_segment_and_joint_kwargs(
        agg_kwargs, {"average": True, "ignore_tolerance": False}
    )

    if score_component == "segment":
        weight = scoring_params["segment"]["score_weight"]
        tolerance = scoring_params["segment"]["score_tolerance"]
        average = agg_kwargs["segment"]["average"]
        ignore_indiv_tolerances = agg_kwargs["segment"]["ignore_tolerance"]
    else:
        weight = {k: v["score_weight"] for k, v in scoring_params.items()}
        tolerance = {k: v["score_tolerance"] for k, v in scoring_params.items()}
        average = {k: v["average"] for k, v in agg_kwargs.items()}
        ignore_indiv_tolerances = {
            k: v["ignore_tolerance"] for k, v in agg_kwargs.items()
        }

    score_histories = {}
    removal_binaries = {}
    for rb_name, rb in rigid_bodies.items():
        rb_trajs: dict[str, MarkerTrajectory] = {
            m: traj for m, traj in marker_trajectories.items() if m in rb.get_markers()
        }

        removal_binaries[rb_name] = []
        score_histories[rb_name] = []

        for frame in range(0, len(frames)):
            nodes = [traj.generate_node(m, frame) for m, traj in rb_trajs.items()]

            current_rigid_body = copy.deepcopy(rb)
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
                score_tolerance=tolerance,
                score_weight=weight,
                aggregate_using_average=average,
                ignore_residual_tolerances=ignore_indiv_tolerances,
                return_composite_score=True,
            )
            worst_nodes = rigid_body_scorer.sort_marker_scores(
                node_scores, threshold=0, max_markers=1
            )

            removals, scores = {}, {}
            for m in current_rigid_body.get_markers():
                removals[m] = False
                scores[m] = 0

            i = 0
            while worst_nodes and i < max_removals_per_frame:
                node, score = worst_nodes[0]
                removals[node] = True
                scores[node] = score if isinstance(score, (float, int)) else score[0]

                current_rigid_body.remove_node(node)
                node_scores = rigid_body_scorer.score_rigid_body_components(
                    current_rigid_body,
                    score_component,
                    residual_type,
                    score_tolerance=tolerance,
                    score_weight=weight,
                    aggregate_using_average=average,
                    ignore_residual_tolerances=ignore_indiv_tolerances,
                    return_composite_score=True,
                )
                worst_nodes = rigid_body_scorer.sort_marker_scores(
                    node_scores, threshold=0, max_markers=1
                )
                i += 1

            removal_binaries[rb_name].append(list(removals.values()))
            score_histories[rb_name].append(list(scores.values()))

    return removal_binaries, score_histories


def get_marker_removals_from_prior_bodies(
    calibrated_rigid_bodies: dict[str, RigidBody],
    marker_trajectories: dict[str, MarkerTrajectory],
    segments_only: bool,
    scoring_params: dict = None,
    agg_kwargs: dict = None,
    max_removals_per_frame: int = 3,
    tolerance_factor: Union[float, dict] = 0.5,
):
    """!Get marker removals from prior bodies.

    Calculates the worst markers per frame and removes them from the rigid body.
    Returns a dictionary of binary arrays indicating which markers were removed per frame.
    Also returns a dictionary of scores for each marker per frame.

    @param calibrated_rigid_bodies dict of initial or calibrated RigidBody instances.
    @param marker_trajectories dict of MarkerTrajectory instances.
    @param segments_only If True, only segment residuals are considered.
    @param scoring_params dict of kwargs for scoring_rigid_body_components. (i.e. score_tolerance, score_weight)
    @param agg_kwargs dict of kwargs for aggregate_residuals. (i.e. average, ignore_tolerance)
    @param max_removals_per_frame Maximum number of markers to remove per frame.
    @param tolerance_factor Factor to reduce segment and joint tolerances by (relative to calibration).
                Can also provide a dictionary of factors for each body.

    """
    residual_type = "prior"
    score_component = "segment" if segments_only else "node"

    scoring_params = rigid_body_scorer.validate_segment_and_joint_kwargs(
        scoring_params, {"score_tolerance": 0, "score_weight": 1}
    )

    agg_kwargs = rigid_body_scorer.validate_segment_and_joint_kwargs(
        agg_kwargs, {"average": True, "ignore_tolerance": False}
    )

    if score_component == "segment":
        weight = scoring_params["segment"]["score_weight"]
        tolerance = scoring_params["segment"]["score_tolerance"]
        average = agg_kwargs["segment"]["average"]
        ignore_indiv_tolerances = agg_kwargs["segment"]["ignore_tolerance"]
    else:
        weight = {k: v["score_weight"] for k, v in scoring_params.items()}
        tolerance = {k: v["score_tolerance"] for k, v in scoring_params.items()}
        average = {k: v["average"] for k, v in agg_kwargs.items()}
        ignore_indiv_tolerances = {
            k: v["ignore_tolerance"] for k, v in agg_kwargs.items()
        }

    score_histories = {}
    removal_binaries = {}
    for rb_name, rb in calibrated_rigid_bodies.items():

        rb_trajs: dict[str, MarkerTrajectory] = {
            m: traj for m, traj in marker_trajectories.items() if m in rb.get_markers()
        }

        removal_binaries[rb_name] = []
        score_histories[rb_name] = []

        tol_factor = (
            tolerance_factor.get(rb_name, 0.5)
            if isinstance(tolerance_factor, dict)
            else tolerance_factor
        )
        initial_rb = copy.deepcopy(rb)
        seg_tols = {
            seg: tol * tol_factor
            for seg, tol in initial_rb.get_segment_tolerances().items()
        }
        initial_rb.update_segment_tolerances(seg_tols)
        ref_lengths = {str(seg): seg.length for seg in rb.segments}

        if not segments_only:
            joint_tols = {
                joint: tol * tol_factor
                for joint, tol in initial_rb.get_joint_tolerances().items()
            }
            initial_rb.update_joint_tolerances(joint_tols)

        prior_rigid_body = None

        for frame in range(0, len(frames)):
            nodes = [traj.generate_node(m, frame) for m, traj in rb_trajs.items()]
            if prior_rigid_body is None:
                current_rigid_body = copy.deepcopy(initial_rb)
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

            node_scores = rigid_body_scorer.score_rigid_body_components(
                current_rigid_body,
                score_component,
                residual_type,
                score_tolerance=tolerance,
                score_weight=weight,
                aggregate_using_average=average,
                ignore_residual_tolerances=ignore_indiv_tolerances,
                return_composite_score=True,
            )
            worst_nodes = rigid_body_scorer.sort_marker_scores(
                node_scores, threshold=0, max_markers=1
            )

            removals, scores = {}, {}
            for m in current_rigid_body.get_markers():
                removals[m] = False
                scores[m] = 0

            i = 0
            while worst_nodes and i < max_removals_per_frame:
                node, score = worst_nodes[0]
                removals[node] = True
                scores[node] = score if isinstance(score, (float, int)) else score[0]

                current_rigid_body.remove_node(node)
                node_scores = rigid_body_scorer.score_rigid_body_components(
                    current_rigid_body,
                    score_component,
                    residual_type,
                    score_tolerance=tolerance,
                    score_weight=weight,
                    aggregate_using_average=average,
                    ignore_residual_tolerances=ignore_indiv_tolerances,
                    return_composite_score=True,
                )
                worst_nodes = rigid_body_scorer.sort_marker_scores(
                    node_scores, threshold=0, max_markers=1
                )
                i += 1

            prior_rigid_body = current_rigid_body

            removal_binaries[rb_name].append(list(removals.values()))
            score_histories[rb_name].append(list(scores.values()))

    return removal_binaries, score_histories


def plot_residual_histories(
    rigid_bodies: dict[str, RigidBody],
    marker_trajectories: dict[str, MarkerTrajectory],
    res_type: Literal["calib", "prior"],
    segments_only: bool,
    agg_kwargs: dict = None,
) -> tuple[plt.Figure, list[plt.Axes]]:
    """!Generates and plots residual histories for rigid bodies.

    WARNING: this is a time-consuming operation.

    @param rigid_bodies dict of RigidBody instances.
    @param marker_trajectories dict of MarkerTrajectory instances.
    @param res_type Residual type to plot. Either "calib" or "prior".
    @param segments_only If True, only plot segment residuals.
    @param agg_kwargs dict of kwargs for aggregate_residuals. (i.e. average, ignore_tolerance)

    """
    LOGGER.info("Generating residuals...")
    residual_histories = rigid_body_scorer.generate_residual_histories(
        rigid_bodies,
        marker_trajectories,
        res_type,
        segments_only,
        agg_kwargs=agg_kwargs,
    )
    LOGGER.info("Plotting residuals histories.")

    nrow = 1 if segments_only else 2
    ncol = len(rigid_bodies)
    fig, ax = plt.subplots(nrow, ncol, figsize=(5 * ncol, 4 * nrow), sharey="row")

    seg_axs = ax if nrow == 1 else ax[0, :]
    joint_axs = [] if segments_only else ax[1, :]

    for i, (body_name, body) in enumerate(rigid_bodies.items()):
        rf_seg = np.array(residual_histories[body_name]["segment"])
        seg_axs[i].plot(rf_seg)
        seg_axs[i].legend(body.get_markers())
        seg_axs[i].set_title(body_name)
        seg_axs[i].set_ylabel("Segment Residual")

        if len(joint_axs) > 0:
            rf_joint = np.array(residual_histories[body_name]["joint"])
            joint_axs[i].plot(rf_joint)
            joint_axs[i].set_ylabel("Joint Residual")

    fig.tight_layout()
    return fig, ax


def plot_removals(
    rigid_bodies: dict[str, RigidBody],
    removal_binaries: dict[str, Union[np.array, list]],
) -> tuple[plt.Figure, list[plt.Axes]]:
    """!Plot marker removals.

    @param rigid_bodies dict of RigidBody instances.
    @param removal_binaries dict of binary arrays or lists of marker removals.
    """
    ncols = len(removal_binaries)
    nrows = max([len(rb.get_markers()) for rb in rigid_bodies.values()])

    fig, ax = plt.subplots(
        nrows, ncols, figsize=(5 * ncols, max(1, nrows)), sharex=True, sharey=True
    )

    for i, (rb_name, rb) in enumerate(rigid_bodies.items()):
        binaries = np.array(removal_binaries[rb_name])
        for j, marker in enumerate(rb.get_markers()):
            arr = binaries[:, j]
            l, g = plot_utils.get_binary_signal_edges(arr)
            plot_utils.draw_shaded_regions(ax[j, i], l, g, color="blue", alpha=0.8)
            ax[j, i].set_title(marker)

            ax[j, i].set_yticks([])

    fig.tight_layout()

    return fig, ax


def plot_scores(
    rigid_bodies: dict[str, RigidBody], score_histories: dict[str, list]
) -> tuple[plt.Figure, list[plt.Axes]]:
    """!Plot marker scores.

    @param rigid_bodies dict of RigidBody instances.
    @param score_histories dict of lists of marker scores.
    """

    ncols = len(score_histories)
    nrows = 1

    fig, ax = plt.subplots(
        nrows, ncols, figsize=(5 * ncols, 4), sharex=True, sharey=True
    )

    for i, (rb_name, rb) in enumerate(rigid_bodies.items()):
        scores = np.array(score_histories[rb_name])
        ax[i].plot(scores)
        ax[i].set_title(rb_name)
        ax[i].set_ylabel("Score")
        ax[i].set_xlabel("Frame Index")
        ax[i].legend(rb.get_markers())

        threshold = 0
        ax[i].axhline(threshold, color="black", linestyle="--")

    fig.tight_layout()

    return fig, ax


def configure_parser():
    """!Configure the argument parser."""
    parser = argparse.ArgumentParser(
        description="Unlabel rigid body markers based on residual scores."
    )

    parser.add_argument(
        "--project_dir",
        type=str,
        default="",
        help="Path to the project directory. Must be provided if using 'offline' mode.",
    )

    parser.add_argument(
        "--trial_name",
        type=str,
        default="",
        help="Name of the trial. Used to fetch vsk. Must be provided if using 'offline' mode.",
    )

    parser.add_argument(
        "--subject_name",
        type=str,
        default="",
        help="Name of the subject. Must be provided if using 'offline' mode.",
    )

    parser.add_argument(
        "--scoring_name",
        type=str,
        default="default",
        help="Name of the scoring configuration to use. (saved to mocap_popy/scripts/unlabel_rb_markers/scoring)",
    )

    parser.add_argument(
        "--offline",
        action="store_true",
        help="Processes c3d files in offline mode and stores removals in a json file.",
    )

    parser.add_argument(
        "--segments_only",
        action="store_true",
        help="Only consider segment residuals.",
    )

    parser.add_argument(
        "--plot_residuals",
        action="store_true",
        help="Plot residuals histories (before scoring).",
    )

    parser.add_argument(
        "--plot_removals",
        action="store_true",
        help="Plot marker removals.",
    )

    parser.add_argument(
        "--plot_scores",
        action="store_true",
        help="Plot marker scores.",
    )

    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Increase logging verbosity to debug.",
    )

    parser.add_argument(
        "--log",
        action="store_true",
        help="Log output to file.",
    )

    parser.add_argument(
        "--no_console",
        action="store_true",
        help="Do not log to console.",
    )

    return parser


def main():

    parser = configure_parser()
    args = parser.parse_args()

    offline = args.offline
    if offline:
        if not args.project_dir or not args.trial_name or not args.subject_name:
            LOGGER.error(
                "Project directory, trial name, and subject name must be provided in offline mode."
            )
            exit()
    pass


# %%
DATASET_DIR = os.path.join(directory.DATASET_DIR, "shoe_stepping")
TRIAL_FN = "trial01.c3d"
SUBJECT_NAME = "subject"

c3d_fp = os.path.join(DATASET_DIR, TRIAL_FN)
vsk_fp = os.path.join(DATASET_DIR, f"{SUBJECT_NAME}.vsk")

calibrated_rigid_bodies = rigid_body_loader.get_rigid_bodies_from_vsk(vsk_fp)

# %%
c3d_reader = c3d_parser.get_reader(c3d_fp)
marker_trajectories = c3d_parser.get_marker_trajectories(c3d_reader)
frames = c3d_parser.get_frames(c3d_reader)

# %%

segments_only = True
# plot_residuals = False

start = time.time()
segments_only = True

scoring = {
    "segment": {"score_tolerance": 0, "score_weight": 1},
    "joint": {"score_tolerance": 0, "score_weight": 0.002778},
}

removal_histories, score_histories = get_marker_removals_from_calibration_bodies(
    calibrated_rigid_bodies,
    marker_trajectories,
    segments_only,
    scoring_params=scoring,
)

print(f"Time: {time.time() - start}")

# %%

fig, ax = plot_removals(calibrated_rigid_bodies, removal_histories)
fig.show()
# %%

fig, ax = plot_scores(calibrated_rigid_bodies, score_histories)
fig.show()

# %%

fig, ax = plot_residual_histories(
    calibrated_rigid_bodies, marker_trajectories, "calib", segments_only
)
