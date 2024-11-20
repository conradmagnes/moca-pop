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
    From scripts directory:
    1. Offline Mode:
        python unassign_rb_markers.py -off -p <project_dir> -tn <trial_name> -sn <subject_name> -start <start_frame> -end <end_frame>

    2. Online Mode:
        python unassign_rb_markers.py -on -sn <subject_name> --start_frame <start_frame> --end end_frame <end_frame>

    Options:
    --------
    Run 'python unassign_rb_markers.py -h' for options.

    Returns:
    --------
    0 if successful, -1 if an error occurred.


    @author C. McCarthy
"""

import argparse
import copy
import logging
import os
import sys
import time
from typing import Literal, Union
import tqdm


import matplotlib.pyplot as plt
import numpy as np
import pydantic

import mocap_popy.config.directory as directory
import mocap_popy.config.logger as logger
from mocap_popy.models.rigid_body import RigidBody, Node
from mocap_popy.models.marker_trajectory import MarkerTrajectory
from mocap_popy.scripts.unassign_rb_markers.scoring import scorer, scoringParameters
from mocap_popy.utils import (
    rigid_body_loader,
    c3d_parser,
    model_template_loader,
)
from mocap_popy.utils import plot_utils, json_utils, dist_utils, hmi


LOGGER = logging.getLogger("UnassignRigidBodyMarkers")


def get_marker_removals_from_calibration_bodies(
    rigid_bodies: dict[str, RigidBody],
    marker_trajectories: dict[str, MarkerTrajectory],
    frames: list[int],
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
                node_scores, threshold=0, max_markers=1, include_duplicates=False
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
                    node_scores, threshold=0, max_markers=1
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
                node_scores = scorer.score_rigid_body_components(
                    current_rigid_body,
                    score_component,
                    residual_type,
                    scoring_parameters=scoring_params,
                    return_composite_score=True,
                )
                worst_nodes = scorer.sort_marker_scores(
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
            arr = binaries[:, j]
            l, g = plot_utils.get_binary_signal_edges(arr)
            if start_frame > 0:
                l = [x + start_frame for x in l]
                g = [x + start_frame for x in g]
            plot_utils.draw_shaded_regions(ax[j, i], l, g, color="blue", alpha=0.8)

            ax[j, i].set_xlim(start_frame, start_frame + len(arr))
            ax[j, i].set_title(marker)

            ax[j, i].set_yticks([])

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


def validate_offline_args(args) -> tuple:
    """!Validate offline mode arguments.
    Mainly checks file paths.

    @return tuple of project_dir, trial_fp, vsk_fp, subject_name
    """

    if not args.project_dir or not args.trial_name or not args.subject_name:
        LOGGER.error(
            "Project directory, trial name, and subject name must be provided in offline mode."
        )
        exit(-1)

    if "example_datasets" in args.project_dir:
        project_dir = os.path.join(
            directory.DATASET_DIR, args.project_dir.split("/")[-1]
        )
    elif args.project_dir in os.listdir(directory.DATASET_DIR):
        LOGGER.info("Found project directory in example datasets.")
        project_dir = os.path.join(directory.DATASET_DIR, args.project_dir)
    else:
        project_dir = args.project_dir

    if not os.path.isdir(project_dir):
        LOGGER.error(f"Project directory does not exist ({project_dir}). Exiting.")
        exit(-1)

    trial_fp = os.path.join(project_dir, f"{args.trial_name}.c3d")
    if not os.path.isfile(trial_fp):
        LOGGER.error(f"Trial file does not exist ({trial_fp}). Exiting.")
        exit(-1)

    vsk_fp = os.path.join(project_dir, f"{args.subject_name}.vsk")
    if not os.path.isfile(vsk_fp):
        LOGGER.error(f"VSK file was not found: {vsk_fp}. Exiting.")
        exit(-1)

    return project_dir, trial_fp, vsk_fp, args.subject_name


def validate_online_args(args, vicon):
    """!Validate online mode arguments.

    Checks loaded trial and subject names, and vsk file.

    @param vicon ViconNexus instance.
    @return tuple of project_dir, trial_fp, vsk_fp, subject_name
    """
    project_dir, trial_name = vicon.GetTrialName()
    if not trial_name:
        LOGGER.error("Load a trial in Nexus before running 'online' mode.")
        exit(-1)

    trial_fp = os.path.join(project_dir, f"{trial_name}.c3d")

    subject_names, subject_templates, subject_statuses = vicon.GetSubjectInfo()

    if not args.subject_name:
        LOGGER.info("Searching for available subject templates...")
        candidate_subject_names = []
        mapper = model_template_loader.load_mapper()
        for name, template, status in zip(
            subject_names, subject_templates, subject_statuses
        ):
            if template in mapper:
                candidate_subject_names.append((name, status))
        if len(candidate_subject_names) == 0:
            LOGGER.error("No matching templates found for trial subjects. Exiting.")
            exit(-1)
        elif len(candidate_subject_names) > 1:
            candidate_subject_names = [
                sn for sn, status in candidate_subject_names if status
            ]
            if len(candidate_subject_names) > 1 or len(candidate_subject_names) == 0:
                LOGGER.error("Could not infer subject name (multiple or none active).")
                exit(-1)

        subject_name, _ = candidate_subject_names[0]

    elif args.subject_name not in subject_names:
        LOGGER.error(f"Subject name '{args.subject_name}' not found in Nexus. Exiting.")
        exit(-1)
    else:
        subject_name = args.subject_name

    vsk_fp = os.path.join(project_dir, f"{subject_name}.vsk")
    if not os.path.isfile(vsk_fp):
        LOGGER.error(f"VSK file was not found: {vsk_fp}. Exiting.")
        exit(-1)

    return project_dir, trial_fp, vsk_fp, subject_name


def validate_start_end_frames(args, frames: list[int]) -> tuple[int, int]:
    """!Validate start and end frames.

    @param args argparse.Namespace instance.
    @param frames list of frame indices.
    @return tuple of start_frame, end_frame
    """
    first_frame = frames[0]
    last_frame = frames[-1]

    if args.start_frame >= last_frame:
        LOGGER.error(
            f"Start frame {args.start_frame} is out of range ({first_frame}-{last_frame}). Exiting."
        )
        exit(-1)

    start_frame = first_frame if args.start_frame < 0 else args.start_frame

    if args.end_frame < 0:
        end_frame = last_frame
    elif args.end_frame < start_frame:
        LOGGER.error(f"End frame {args.end_frame} is before start frame. Exiting.")
        exit(-1)
    elif args.end_frame >= last_frame:
        LOGGER.warning(
            f"End frame {args.end_frame} is out of range ({first_frame}-{last_frame}). Setting to end."
        )
        end_frame = last_frame
    else:
        end_frame = args.end_frame

    return start_frame, end_frame


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


def ask_user_to_confirm_removals(max_tries: int = 5) -> bool:
    """!Ask the user to confirm removals.

    Exits if user enters 'q', 'quit', or 'exit', or if max_tries is reached.
    """
    ui = ""
    i = 0
    while ui.lower() not in ("y", "yes", "n", "no") and i < max_tries:
        ui = hmi.get_user_input(
            "Would you like to remove marker labels from the trial? (y/n): "
        )
        if ui.lower() in ("n", "no"):
            return False
        elif ui.lower() in ("y", "yes"):
            return True
        elif ui.lower() in ("q", "quit", "exit"):
            LOGGER.info("Exit key pressed. Exiting.")
            exit(0)
        else:
            LOGGER.warning("Invalid input. Please enter 'y' or 'n'.")

        i += 1
        if i == max_tries:
            LOGGER.warning("Input limit reached. Continuing without removal.")

    return False


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
        "--force",
        action="store_true",
        help="Force continue through user prompts (i.e. continue after plotting, remove markers)",
    )

    parser.add_argument(
        "-s",
        "--save_to_file",
        action="store_true",
        help="Saves the results to a file, i.e. a record of frames each marker was removed from.",
    )

    parser.add_argument(
        "-off",
        "--offline",
        action="store_true",
        help="Processes c3d files offline rather than connected to Nexus. Markers will not be removed from the trial.",
    )

    parser.add_argument(
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
        "--segments_only",
        action="store_true",
        help="Only score segment length residuals (ignore joint angles). Significantly reduces computation time.",
    )

    parser.add_argument(
        "-p",
        "--project_dir",
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
        help="Name of the scoring configuration to use. (saved to mocap_popy/scripts/unassign_rb_markers/scoring)",
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

    if args.console and not args._new_console_opened:
        dist_utils.run_in_new_console(
            close_console_on_exit=(not args.keep_console_open)
        )

    mode = "w" if args.log else "off"
    logger.set_root_logger(name="unassign_rb_markers", mode=mode)

    if args.verbose:
        LOGGER.setLevel(logging.DEBUG)

    LOGGER.info("Running `unassign_rb_markers.py` ...")
    LOGGER.debug(args)

    ## Validate Args
    offline = args.offline
    if offline:
        vicon = None
        project_dir, trial_fp, vsk_fp, subject_name = validate_offline_args(args)
    else:
        try:
            from viconnexusapi import ViconNexus
        except ImportError as e:
            LOGGER.error(f"Error importing Vicon Nexus. Check installation. {e}")
            exit(-1)

        vicon = ViconNexus.ViconNexus()
        project_dir, trial_fp, vsk_fp, subject_name = validate_online_args(args, vicon)

    segments_only = args.segments_only
    residual_type = args.residual_type

    ## Load Data
    scoring_params = load_scoring_parameters(args.scoring_name)

    LOGGER.debug(f"Scoring Params: {scoring_params.model_dump_json()}")

    calibrated_rigid_bodies = rigid_body_loader.get_rigid_bodies_from_vsk(vsk_fp)

    if offline:
        c3d_reader = c3d_parser.get_reader(trial_fp)
        marker_trajectories = c3d_parser.get_marker_trajectories(c3d_reader)
        trial_frames = c3d_parser.get_frames(c3d_reader)
    else:
        markers = vicon.GetMarkerNames(subject_name)
        marker_trajectories = {}
        for m in markers:
            x, y, z, e = vicon.GetTrajectory(subject_name, m)
            marker_trajectories[m] = MarkerTrajectory(x, y, z, e)

        trial_range = vicon.GetTrialRange()
        trial_frames = list(range(trial_range[0], trial_range[1]))

    start, end = validate_start_end_frames(args, trial_frames)
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
        )
    else:
        removal_histories, score_histories = (
            get_marker_removals_from_calibration_bodies(
                calibrated_rigid_bodies,
                marker_trajectories,
                frames,
                segments_only,
                scoring_params=scoring_params,
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
        plt.show(block=offline)

    ## Remove Markers from Online Trial
    if not offline and ask_user_to_confirm_removals(max_tries=5):
        remove_markers_from_online_trial(
            vicon, subject_name, removal_histories, calibrated_rigid_bodies, start
        )

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

        timestamp = time.strftime("%Y%m%d-%H%M%S")
        file_ext = "txt" if args.output_file_type == "txt" else "json"
        output_fp = os.path.join(project_dir, f"removals_{timestamp}.{file_ext}")
        LOGGER.info("Writing removal ranges to file.")
        write_removal_ranges_to_file(removal_ranges, output_fp)

    ## Cleanup
    LOGGER.info("Done!")

    exit(0)


def test_main_with_args():
    """!Test main function with args. Useful for debugging."""

    sys.argv = [
        "unassign_rb_markers.py",
        "-v",
        # "-s",
        # "--offline",
        # "--output_file_type",
        # "txt",
        # "--project_dir",
        # "shoe_stepping",
        # "--trial_name",
        # "trial01",
        # "--subject_name",
        # "subject",
        "--residual_type",
        "calib",
        # "--segments_only",
        "--start_frame",
        "10650",
        "--end_frame",
        "10750",
        "--plot_residuals",
        "--plot_removals",
        "--plot_scores",
    ]

    main()


if __name__ == "__main__":
    # print("d")
    test_main_with_args()
    # main()
