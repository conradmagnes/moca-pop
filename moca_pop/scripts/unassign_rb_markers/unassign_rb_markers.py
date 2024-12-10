"""!

    Unassign Rigid Body Markers
    ===========================

    This script uses the RigidBody model to score and unassign labeled markers from a trial based on residual scores.
    Residuals can be calculated based on calibration or prior frames.
    The script can be run 'online', using an active trial in Vicon Nexus, or 'offline',
    using a specified project directory with a C3D and VSK file.

    The script will optionally plot the calculated marker residuals, scroes, and unassignments (removals).
    The results can be saved to a file (txt or json).

    Scoring Strategies:
    -------------------
    This script includes several configuration options to aggregate and weight residuals to determine which markers to remove.
    These parameters can be set in a JSON file in the 'scoring' directory. The 'default.json' file is used if no other file is provided.

    Segment residuals are the relative difference between measured and calibrated (or prior) segment lengths.
    Joint residuals are the absolute difference between measured and calibrated (or prior) joint angles.
    Calculating joint residuals is computationally expensive and may not be necessary for all use cases. 
    The optional --segments_only flag can be used to only consider segment residuals.
    The --plot_residuals flag can be used to visualize the segment and joint residuals for each marker.

    Thresolding and aggregation occurs at the segment, joint and/or node level.
    Thresholds (or tolerances) for individual segments and joints should be set in the model template file.
    When aggregating segment and joint residuals for each marker, any residual within (+/-) the tolerance is set to zero.
    Aggregation for segments and joints can be set separately to a sum or average for each node.
    Aggregated segment and joint residuals are then weighted and summed to determine the final score for each node.

    An additional threshold can be set for each (or all) node scores in the scoring configuration file. If none is provided, the default is 0.
    
    The script removes from the rigid body the marker with the highest score, recalculates residuals, and repeats this process for a specified number of iterations.
    The removals and scores are saved and can be plotted to visualize the process.
    
    Usage:
    ------
    python unassign_rb_markers.py [-h] [-v] [-l] [-c] [-f] [-p] [-s] [-i] [-off] [-out {json,txt}] [-res {calib,prior}] [-pn PROJECT_NAME] [-tn TRIAL_NAME] [-sn SUBJECT_NAME]
                              [--scoring_name SCORING_NAME] [--include_rbs INCLUDE_RBS] [--exclude_rbs EXCLUDE_RBS] [--custom_rbs CUSTOM_RBS] [--segments_only]
                              [--max_removals_per_frame MAX_REMOVALS_PER_FRAME] [--start_frame START_FRAME] [--end_frame END_FRAME] [--plot_residuals] [--plot_scores] [--plot_removals]
                              [--keep_console_open]

    Examples:
        python unassign_rb_markers.py -v -l -i -off -res "calib" -pn "shoe_stepping" -tn "trial01" -sn "subject" --scoring_name "foot" --segments_only --plot_removals
                              
    Run 'python unassign_rb_markers.py -h' for detailed option descriptions.

    Returns:
    --------
    0 if successful, -1 if an error occurred.

    @author C. McCarthy
"""

import argparse
import copy
import dash
import logging
import os
import sys
from typing import Literal, Union
import tqdm

import matplotlib.pyplot as plt
import numpy as np
import pydantic

import moca_pop.config.directory as directory
import moca_pop.config.logger as logger
from moca_pop.models.rigid_body import RigidBody, Node
from moca_pop.models.marker_trajectory import MarkerTrajectory
from moca_pop.scripts.unassign_rb_markers.scoring import scorer, scoringParameters
from moca_pop.utils import rigid_body_loader, c3d_parser, vicon_utils
from moca_pop.utils import plot_utils, json_utils, dist_utils, hmi, argparse_utils

import moca_pop.aux_scripts.interactive_score_analyzer.app as isa


LOGGER = logging.getLogger("UnassignRigidBodyMarkers")


def get_marker_removals_from_calibration_bodies(
    rigid_bodies: dict[str, RigidBody],
    marker_trajectories: dict[str, MarkerTrajectory],
    frames: list[int],
    segments_only: bool,
    scoring_params: scoringParameters.ScoringParameters = None,
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

    scoring_params = scoring_params or scoringParameters.ScoringParameters()
    LOGGER.debug(f"Scoring Params Used: {scoring_params}")

    score_histories = {}
    removal_binaries = {}
    for rb_name, rb in rigid_bodies.items():
        LOGGER.info(f"Processing rigid body: {rb_name}")
        rb_trajs: dict[str, MarkerTrajectory] = {
            m: traj for m, traj in marker_trajectories.items() if m in rb.get_markers()
        }

        removal_binaries[rb_name] = []
        score_histories[rb_name] = []

        for frame in tqdm.tqdm(frames, desc="Iterating through frames"):
            nodes = [traj.generate_node(m, frame) for m, traj in rb_trajs.items()]

            current_rigid_body = copy.deepcopy(rb)
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
                scoring_parameters=scoring_params,
                return_composite_score=True,
            )
            worst_nodes = scorer.sort_marker_scores(
                node_scores,
                threshold=scoring_params.removal_threshold,
                max_markers=1,
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
                node_scores = scorer.score_rigid_body_components(
                    current_rigid_body,
                    score_component,
                    residual_type,
                    scoring_parameters=scoring_params,
                    return_composite_score=True,
                )
                worst_nodes = scorer.sort_marker_scores(
                    node_scores,
                    threshold=scoring_params.removal_threshold,
                    max_markers=1,
                )
                i += 1

            removal_binaries[rb_name].append(list(removals.values()))
            score_histories[rb_name].append(list(scores.values()))

    return removal_binaries, score_histories


def get_marker_removals_from_prior_bodies(
    calibrated_rigid_bodies: dict[str, RigidBody],
    marker_trajectories: dict[str, MarkerTrajectory],
    frames: list[int],
    segments_only: bool,
    scoring_params: scoringParameters.ScoringParameters,
    max_removals_per_frame: int = 3,
    tolerance_factor: Union[float, dict] = 1,
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

    scoring_params = scoring_params or scoringParameters.ScoringParameters()
    LOGGER.debug(f"Scoring Params Used: {scoring_params}")

    score_histories = {}
    removal_binaries = {}
    for rb_name, rb in calibrated_rigid_bodies.items():
        LOGGER.info(f"Processing rigid body: {rb_name}")

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

        for frame in tqdm.tqdm(frames, desc="Iterating through frames"):
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

            node_scores = scorer.score_rigid_body_components(
                current_rigid_body,
                score_component,
                residual_type,
                scoring_parameters=scoring_params,
                return_composite_score=True,
            )
            worst_nodes = scorer.sort_marker_scores(
                node_scores,
                threshold=scoring_params.removal_threshold,
                max_markers=1,
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
                node_scores = scorer.score_rigid_body_components(
                    current_rigid_body,
                    score_component,
                    residual_type,
                    scoring_parameters=scoring_params,
                    return_composite_score=True,
                )
                worst_nodes = scorer.sort_marker_scores(
                    node_scores,
                    threshold=scoring_params.removal_threshold,
                    max_markers=1,
                )
                i += 1

            prior_rigid_body = current_rigid_body

            removal_binaries[rb_name].append(list(removals.values()))
            score_histories[rb_name].append(list(scores.values()))

    return removal_binaries, score_histories


def plot_residual_histories(
    rigid_bodies: dict[str, RigidBody],
    marker_trajectories: dict[str, MarkerTrajectory],
    frames: list[int],
    res_type: Literal["calib", "prior"],
    segments_only: bool,
    plot_frames: list[int] = None,
    agg_params: scoringParameters.ScoringParameters = None,
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
    residual_histories = {}
    for rb_name, rb in rigid_bodies.items():
        residual_histories[rb_name] = scorer.generate_residual_histories(
            rb,
            marker_trajectories,
            res_type,
            segments_only,
            start_frame=frames[0],
            end_frame=frames[-1],
            agg_params=agg_params,
        )

    LOGGER.info("Plotting residuals histories.")

    nrow = 1 if segments_only else 2
    ncol = len(rigid_bodies)
    fig, ax = plt.subplots(nrow, ncol, figsize=(4 * ncol, (3 * nrow)), sharey="row")

    seg_axs = ax if nrow == 1 else ax[0, :]
    joint_axs = [] if segments_only else ax[1, :]

    x = plot_frames if plot_frames and len(plot_frames) == len(frames) else frames

    for i, (body_name, body) in enumerate(rigid_bodies.items()):
        rf_seg = np.array(residual_histories[body_name]["segment"])
        for marker_idx in range(rf_seg.shape[1]):
            seg_axs[i].plot(x, rf_seg[:, marker_idx])
        seg_axs[i].legend(body.get_markers())
        seg_axs[i].set_title(body_name)
        seg_axs[i].set_ylabel("Segment Residual")

        if len(joint_axs) > 0:
            rf_joint = np.array(residual_histories[body_name]["joint"])
            for joint_idx in range(rf_joint.shape[1]):
                joint_axs[i].plot(x, rf_joint[:, joint_idx])
            joint_axs[i].set_ylabel("Joint Residual")

    fig.tight_layout()
    return fig, ax


def plot_removals(
    rigid_bodies: dict[str, RigidBody],
    removal_binaries: dict[str, Union[np.array, list]],
    start_frame: int = 0,
) -> tuple[plt.Figure, list[plt.Axes]]:
    """!Plot marker removals.

    @param rigid_bodies dict of RigidBody instances.
    @param removal_binaries dict of binary arrays or lists of marker removals.
    """
    ncols = len(removal_binaries)
    nrows = max([len(rb.get_markers()) for rb in rigid_bodies.values()])

    fig, ax = plt.subplots(
        nrows,
        ncols,
        figsize=(5 * ncols, max(1, int(0.75 * nrows))),
        sharex=True,
        sharey=True,
    )

    for i, (rb_name, rb) in enumerate(rigid_bodies.items()):
        binaries = np.array(removal_binaries[rb_name])

        for j, marker in enumerate(rb.get_markers()):
            plt_ax = ax[j, i] if ncols > 1 else ax[j]

            arr = binaries[:, j]
            l, g = plot_utils.get_binary_signal_edges(arr)
            if start_frame > 0:
                l = [x + start_frame for x in l]
                g = [x + start_frame for x in g]
            plot_utils.draw_shaded_regions(plt_ax, l, g, color="blue", alpha=0.8)

            plt_ax.set_xlim(start_frame, start_frame + len(arr))
            plt_ax.set_title(marker)

            plt_ax.set_yticks([])

    fig.tight_layout()

    return fig, ax


def plot_scores(
    rigid_bodies: dict[str, RigidBody],
    score_histories: dict[str, list],
    start_frame: int = 0,
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
        x = np.arange(start_frame, start_frame + len(scores))
        ax[i].plot(x, scores)
        ax[i].set_title(rb_name)
        ax[i].set_ylabel("Score")
        ax[i].set_xlabel("Frame Index")
        ax[i].legend(rb.get_markers())

        threshold = 0
        ax[i].axhline(threshold, color="black", linestyle="--")

    fig.tight_layout()

    return fig, ax


def load_scoring_parameters(name) -> scoringParameters.ScoringParameters:
    scoring_dir = os.path.join(
        directory.SCRIPTS_DIR, "unassign_rb_markers", "scoring", "saved_parameters"
    )
    scoring_fp = os.path.join(scoring_dir, f"{name}.json")

    try:
        params = json_utils.import_json_as_str(scoring_fp)
        return scoringParameters.ScoringParameters.model_validate_json(params)
    except (
        FileNotFoundError,
        json_utils.json.JSONDecodeError,
        pydantic.ValidationError,
    ) as e:
        LOGGER.error(f"Error loading scoring parameters at {scoring_fp}. {e}")
        ui = hmi.get_user_input(
            "Continue with default parameters? (y/n): ",
            exit_on_quit=True,
            choices=[hmi.YES_KEYWORDS, hmi.NO_KEYWORDS],
        )
        if ui == 1:
            exit(0)
        LOGGER.info("Continuing.")

    return scoringParameters.ScoringParameters()


def write_removal_ranges_to_file(removal_ranges: dict, file_path: str):
    """!Write removal ranges to a file."""
    if file_path.endswith(".txt"):
        with open(file_path, "w") as file:
            for body_name, markers in removal_ranges.items():
                file.write(f"{body_name}:\n")
                for marker_name, ranges in markers.items():
                    if ranges:
                        ranges_str = ", ".join([f"{r[0]} - {r[1]}" for r in ranges])
                        file.write(f"  {marker_name} Frames to Remove: {ranges_str}\n")
                    else:
                        file.write(f"  {marker_name} Frames to Remove: None\n")
    else:
        json_utils.export_dict_as_json(removal_ranges, file_path)


def remove_markers_from_online_trial(
    vicon, subject_name, removal_histories, rigid_bodies, start
):
    """!Remove markers from an online trial.

    @param vicon ViconNexus instance.
    @param subject_name Name of the subject.
    @param removal_histories dict of removals. Should be binary arrays indicated 0=keep, 1=remove for each marker.
    @param rigid_bodies dict of RigidBody instances.
    @param start Start frame for removals.
    """

    for rb_name, removals in removal_histories.items():
        total_iters = len(removals) * len(removals[0])
        pbar = tqdm.tqdm(total=total_iters, desc=f"Removing markers from {rb_name}")
        for i, marker in enumerate(rigid_bodies[rb_name].get_markers()):
            arr = np.array(removals)[:, i]
            for frame, remove in enumerate(arr):
                pbar.update(1)
                if not remove:
                    continue
                vframe = int(frame + start)
                x, y, z, _ = vicon.GetTrajectoryAtFrame(subject_name, marker, vframe)
                vicon.SetTrajectoryAtFrame(subject_name, marker, vframe, x, y, z, False)

    LOGGER.info("Markers removed.")


def configure_parser():
    """!Configure the argument parser."""
    parser = argparse.ArgumentParser(
        description="Unassign rigid body markers based on residual scores."
    )

    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Increase logging verbosity to debug.",
    )

    parser.add_argument(
        "-l",
        "--log",
        action="store_true",
        help="Log output to file.",
    )
    parser.add_argument(
        "-c",
        "--console",
        action="store_true",
        help="Opens a new console for logging. Useful for running script in Vicon Nexus.",
    )

    parser.add_argument(
        "-f",
        "--force_true",
        action="store_true",
        help="Force continue through user prompts.",
    )
    parser.add_argument(
        "-p",
        "--preserve_markers",
        action="store_true",
        help="Do not remove markers from the trial.",
    )
    parser.add_argument(
        "-s",
        "--save_to_file",
        action="store_true",
        help="Saves the results to a file, i.e. a record of frames each marker was removed from.",
    )
    parser.add_argument(
        "-i",
        "--inspect",
        action="store_true",
        help="Allows user to select a frame to inspect in the Interactive Score Analyzer.",
    )

    parser.add_argument(
        "-off",
        "--offline",
        action="store_true",
        help="Processes c3d files offline rather than connected to Nexus. Markers will not be removed from the trial.",
    )

    parser.add_argument(
        "-out",
        "--output_file_type",
        type=str,
        default="txt",
        choices=["json", "txt"],
        help="Output file type for removals.",
    )

    parser.add_argument(
        "-res",
        "--residual_type",
        type=str,
        default="calib",
        choices=["calib", "prior"],
        help="Type of residuals to consider for scoring. `calib` compares to calibration, `prior` compares to prior frame.",
    )

    parser.add_argument(
        "-pn",
        "--project_name",
        type=str,
        default="",
        help="Path to the project directory. Must be provided if using 'offline' mode.",
    )

    parser.add_argument(
        "-tn",
        "--trial_name",
        type=str,
        default="",
        help="Name of the trial. Used to fetch data. Must be provided if using 'offline' mode (ignore file ext).",
    )

    parser.add_argument(
        "-sn",
        "--subject_name",
        type=str,
        default="",
        help="Name of the subject. Must be provided if using 'offline' mode.",
    )

    parser.add_argument(
        "--scoring_name",
        type=str,
        default="default",
        help="Name of the scoring configuration to use. (saved to moca_pop/scripts/unassign_rb_markers/scoring)",
    )
    parser.add_argument(
        "--include_rbs",
        type=str,
        default="",
        help="Comma-separated list of rigid bodies to include in processing. (i.e. 'rb1,rb2').",
    )
    parser.add_argument(
        "--exclude_rbs",
        type=str,
        default="",
        help="Comma-separated list of rigid bodies to exclude from processing. (i.e. 'rb1,rb2').",
    )
    parser.add_argument(
        "--custom_rbs",
        type=str,
        default="",
        help="Comma-separated list of custom rigid bodies to process (i.e. bodies that span multiple VSK segments).",
    )

    parser.add_argument(
        "--segments_only",
        action="store_true",
        help="Only score segment length residuals (ignore joint angles). Significantly reduces computation time.",
    )

    parser.add_argument(
        "--max_removals_per_frame",
        type=int,
        default=3,
        help="Maximum number of markers to remove per frame, per rigid body.",
    )

    parser.add_argument(
        "--start_frame",
        type=int,
        default=-1,
        help="Start frame for processing. Default is -1 (start of trial).",
    )
    parser.add_argument(
        "--end_frame",
        type=int,
        default=-1,
        help="End frame for processing. Default is -1 (end of trial).",
    )

    parser.add_argument(
        "--plot_residuals",
        action="store_true",
        help="Plot residuals histories (before scoring).",
    )

    parser.add_argument(
        "--plot_scores",
        action="store_true",
        help="Plot marker scores.",
    )

    parser.add_argument(
        "--plot_removals",
        action="store_true",
        help="Plot marker removals.",
    )

    parser.add_argument(
        "--keep_console_open",
        action="store_true",
        help="Keep console open after script finishes.",
    )

    parser.add_argument(
        "--_new_console_opened",
        action="store_true",
        help=argparse.SUPPRESS,  # Used when new console is opened to avoid duplicate spawns.
    )

    return parser


def main():
    """!Main script execution."""

    ## Parse Args
    parser = configure_parser()
    args = parser.parse_args()

    console = args.console or args.inspect

    if console and not args._new_console_opened:
        dist_utils.run_in_new_console(
            close_console_on_exit=(not args.keep_console_open)
        )

    if args.verbose:
        LOGGER.setLevel(logging.DEBUG)
        scorer.LOGGER.setLevel(logging.DEBUG)

    ## Validate Args
    offline = args.offline
    if offline:
        vicon = None
        project_dir, trial_fp, vsk_fp, subject_name = (
            argparse_utils.validate_offline_paths(
                args.project_name, args.trial_name, args.subject_name
            )
        )
    else:
        try:
            from viconnexusapi import ViconNexus
        except ImportError as e:
            LOGGER.error(f"Error importing Vicon Nexus. Check installation. {e}")
            exit(-1)

        vicon = ViconNexus.ViconNexus()
        project_dir, trial_fp, vsk_fp, subject_name = (
            argparse_utils.validate_online_paths(vicon, args.subject_name)
        )

    mode = "w" if args.log else "off"
    trial_name = trial_fp.split(os.sep)[-1].split(".")[0]
    if mode == "w":
        log_path = os.path.join(project_dir, "logs")
        os.makedirs(log_path, exist_ok=True)
        logger.set_log_dir(log_path)

    logger.set_root_logger(name=f"{trial_name}_unassign_rb_markers", mode=mode)

    LOGGER.info("Running `unassign_rb_markers.py` ...")
    LOGGER.debug(f"Arguments: {args}")

    LOGGER.info(
        "Project: {}, Trial: {}, VSK: {}".format(
            *[os.path.basename(x) for x in [project_dir, trial_fp, vsk_fp]]
        ),
    )
    segments_only = args.segments_only
    residual_type = args.residual_type

    ## Load Data
    scoring_params = load_scoring_parameters(args.scoring_name)

    LOGGER.debug(f"Scoring Params: {scoring_params.model_dump_json()}")

    if args.custom_rbs:
        rbs = args.custom_rbs.split(",")
        LOGGER.debug(f"Loading custom rigid bodies: {rbs}")
        calibrated_rigid_bodies = rigid_body_loader.get_custom_rigid_bodies_from_vsk(
            vsk_fp, rbs
        )
        LOGGER.debug(f"Fetched custom rigid bodies: {calibrated_rigid_bodies.keys()}")
    else:
        include = args.include_rbs.split(",") if args.include_rbs else None
        exclude = args.exclude_rbs.split(",") if args.exclude_rbs else None
        calibrated_rigid_bodies = rigid_body_loader.get_rigid_bodies_from_vsk(
            vsk_fp, include=include, exclude=exclude
        )

    if not calibrated_rigid_bodies:
        LOGGER.error("No rigid bodies found in VSK. Exiting.")
        exit(-1)

    if offline:
        c3d_reader = c3d_parser.get_reader(trial_fp)
        marker_trajectories = c3d_parser.get_marker_trajectories(c3d_reader)
        trial_frames = c3d_parser.get_frames(c3d_reader)
    else:
        marker_trajectories = vicon_utils.get_marker_trajectories(vicon, subject_name)
        trial_frames = vicon_utils.get_trial_frames(vicon)

    start, end = argparse_utils.validate_start_end_frames(
        args.start_frame, args.end_frame, trial_frames
    )
    frames = list(range(start - trial_frames[0], end - trial_frames[0] + 1))

    ## Optional Plot of Residuals
    if args.plot_residuals:
        agg_params = copy.deepcopy(scoring_params)
        agg_params.preagg_thresholding = scoringParameters.PreAggTresholding(
            segment=False, joint=False
        )
        fig, ax = plot_residual_histories(
            calibrated_rigid_bodies,
            marker_trajectories,
            frames,
            residual_type,
            segments_only,
            plot_frames=trial_frames[frames[0] : frames[-1] + 1],
            agg_params=agg_params,
        )
        fig.show()
        LOGGER.info("Keep plot open to continue to see other plots.")

        if args.force_true:
            ui = 0
        else:
            ui = hmi.get_user_input(
                "Continue? (y/n): ",
                exit_on_quit=True,
                choices=[hmi.YES_KEYWORDS, hmi.NO_KEYWORDS],
            )
        if ui == 1:
            LOGGER.info("Done.")
            exit(0)

    # Get Marker Removals
    if residual_type == "prior":
        removal_histories, score_histories = get_marker_removals_from_prior_bodies(
            calibrated_rigid_bodies,
            marker_trajectories,
            frames,
            segments_only,
            scoring_params=scoring_params,
            max_removals_per_frame=args.max_removals_per_frame,
        )
    else:
        removal_histories, score_histories = (
            get_marker_removals_from_calibration_bodies(
                calibrated_rigid_bodies,
                marker_trajectories,
                frames,
                segments_only,
                scoring_params=scoring_params,
                max_removals_per_frame=args.max_removals_per_frame,
            )
        )

    ## Plot Scores and Removals
    if args.plot_scores:
        fig, ax = plot_scores(
            calibrated_rigid_bodies,
            score_histories,
            start_frame=trial_frames[frames[0]],
        )

    if args.plot_removals:
        fig, ax = plot_removals(
            calibrated_rigid_bodies,
            removal_histories,
            start_frame=trial_frames[frames[0]],
        )

    if any([args.plot_removals, args.plot_scores, args.plot_residuals]):
        LOGGER.info("Plotting...")
        block = (offline or args.preserve_markers) and not args.inspect
        plt.show(block=block)

    ## Write Removal Ranges to File
    if args.save_to_file:
        removal_ranges = {}
        for rb_name, removals in removal_histories.items():
            removal_ranges[rb_name] = {}
            for i, marker in enumerate(calibrated_rigid_bodies[rb_name].get_markers()):
                arr = np.array(removals)[:, i]
                l, g = plot_utils.get_binary_signal_edges(arr)
                start_end_list = [[l[i] + start, g[i] + start] for i in range(len(l))]
                removal_ranges[rb_name][marker] = start_end_list

        file_ext = "txt" if args.output_file_type == "txt" else "json"
        tn = trial_fp.split(os.sep)[-1].split(".")[0]
        output_fn = f"{tn}_unassign-results"
        output_fp = directory.get_next_filename(log_path, output_fn, file_ext)
        write_removal_ranges_to_file(removal_ranges, output_fp)
        LOGGER.info(f"Removal ranges written to {output_fp}")

    ## Inspect
    if args.inspect:
        loop_inspector = True
        str_trial_frames = [str(f) for f in trial_frames]
        while loop_inspector:
            ui = hmi.get_validated_user_input(
                "Enter frame to inspect, or 's' to skip: ",
                exit_on_quit=True,
                num_tries=5,
                valid_inputs=["s"] + str_trial_frames,
            )
            if ui.lower() == "s":
                loop_inspector = False
            else:
                isa_args = [
                    "-pn",
                    project_dir,
                    "-sn",
                    subject_name,
                    "-tn",
                    trial_name,
                    "--frame",
                    str(ui),
                    "--scoring_name",
                    args.scoring_name,
                ]
                if offline:
                    isa_args.append("-off")
                if args.verbose:
                    isa_args.append("-v")
                if args.custom_rbs:
                    body_choices = [f"[0] Exit"]
                    body_choices.extend(
                        [
                            f"[{i+1}] {rb}"
                            for i, rb in enumerate(calibrated_rigid_bodies.keys())
                        ]
                    )
                    ui_body = hmi.get_user_input(
                        "\n".join(
                            ["Enter the rigid body to inspect:", *body_choices, "\n"]
                        ),
                        exit_on_quit=True,
                        choices=body_choices,
                    )
                    if ui_body == 0:
                        loop_inspector = False
                        continue

                    isa_args.append("--custom")
                    isa_args.append("--rb_name")
                    isa_args.append(list(calibrated_rigid_bodies.keys())[ui_body - 1])

                LOGGER.info(
                    "Opening Interactive Score Analyzer as a subprocess. This may take a moement to load"
                )
                isa.run_as_subprocess(isa_args)

                LOGGER.info("ISA has shutdown.")

    ## Remove Markers from Online Trial
    if not (offline or args.preserve_markers):
        if args.force_true:
            ui = 0
        else:
            ui = hmi.get_user_input(
                "Remove markers from trial? (y/n): ",
                exit_on_quit=True,
                choices=[hmi.YES_KEYWORDS, hmi.NO_KEYWORDS],
                num_tries=5,
            )
        if ui == 0:
            remove_markers_from_online_trial(
                vicon, subject_name, removal_histories, calibrated_rigid_bodies, start
            )

    ## Cleanup
    LOGGER.info("Done!")

    exit(0)


def test_main_with_args():
    """!Test main function with args. Useful for debugging."""

    sys.argv = [
        "unassign_rb_markers.py",
        "-v",
        # "-s",
        "-f",
        "-p",
        "--scoring_name",
        "foot",
        "--offline",
        # "--output_file_type",
        # "txt",
        "--project_name",
        "shoe_stepping",
        "--trial_name",
        "trial01",
        "--subject_name",
        "subject",
        "--residual_type",
        "calib",
        "--segments_only",
        "--start_frame",
        "5000",
        "--end_frame",
        "6000",
        # "--plot_residuals",
        "--plot_removals",
        # "--plot_scores",
        "--inspect",
    ]

    main()


if __name__ == "__main__":
    # test_main_with_args()
    main()
