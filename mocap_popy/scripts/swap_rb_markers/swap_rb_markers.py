"""!
    Swap RB Markers
    =================

    @author: C. McCarthy
"""

import argparse
import os
import copy
import logging
from typing import Union
import sys

import numpy as np
import tqdm
import matplotlib.pyplot as plt

from mocap_popy.config import directory, logger
import mocap_popy.models.rigid_body as rb
from mocap_popy.models.marker_trajectory import MarkerTrajectory
import mocap_popy.utils.quality_check as qc
from mocap_popy.utils import (
    rigid_body_loader,
    argparse_utils,
    dist_utils,
    json_utils,
    c3d_parser,
    vicon_utils,
    hmi,
)
import mocap_popy.aux_scripts.interactive_score_analyzer.app as isa


LOGGER = logging.getLogger("SwapRBMarkers")


def generate_greedy_search_order(calibrated_body: rb.RigidBody) -> dict[str, list[str]]:
    """!For each marker, get list of other markers in the body in order of increasing distance.

    @param calibrated_body Calibrated rigid body.
    @return Dictionary of marker to list of markers.
    """
    greedy_search_order = {}
    for n in calibrated_body.nodes:
        segments_to_other_nodes = [
            rb.Segment((n, o)) for o in calibrated_body.nodes if o != n
        ]
        for seg in segments_to_other_nodes:
            seg.compute_length()
        sorted_segs = sorted(segments_to_other_nodes, key=lambda s: s.length)
        greedy_search_order[n.marker] = [seg.nodes[1].marker for seg in sorted_segs]
    return greedy_search_order


def generate_swap_candidates(
    fit_body: rb.RigidBody,
    active_body: rb.RigidBody,
    distance_tolerance: Union[str, float],
    greedy_search_order: dict = None,
) -> dict:
    """!Generate a dictionary of candidate swaps based on a radial tolerance.

    If a marker is not within the radial tolerance of its corresponding marker in the fit body,
    the marker is considered a candidate for swapping.

    @param fit_body Calibrated rigid body, (after best fit to the active body).
    @param active_body Active rigid body.
    @param distance_tolerance Radial tolerance for swapping.

    @return Dictionary of candidate swaps.
    """
    greedy_search_order = greedy_search_order or generate_greedy_search_order(fit_body)
    candidate_swaps = {}
    for n in active_body.nodes:
        cn = fit_body.get_node(n.marker)
        if np.linalg.norm(n.position - cn.position) < distance_tolerance:
            continue
        for swap_marker in greedy_search_order[n.marker]:
            cn = fit_body.get_node(swap_marker)
            if np.linalg.norm(n.position - cn.position) < distance_tolerance:
                candidate_swaps[n.marker] = swap_marker
                break
    return candidate_swaps


def validate_swap_candidates(
    candidate_swaps: dict, fit_body: rb.RigidBody, active_body: rb.RigidBody
) -> bool:
    """!
    Validate and refine candidate swaps to ensure markers are correctly assigned.

    If a marker is assigned to multiple markers in the active body, the closest marker is chosen as the target.
    Other assignments are invalidated (removed from the dictionary).

    @param candidate_swaps Dictionary of marker swap candidates (original_marker -> swap_marker).
    @param fit_body Calibrated rigid body.
    @param active_body Active rigid body.

    @return True if all swaps are valid, False otherwise.
    """
    if not candidate_swaps:
        return True

    if set(candidate_swaps.values()) == set(candidate_swaps.keys()):
        # LOGGER.debug("All swaps are valid.")
        return True

    # LOGGER.info("Some swaps are invalid. Attempting to correct.")

    invalid_swaps = []
    shared_targets = [
        target
        for target in candidate_swaps.values()
        if list(candidate_swaps.values()).count(target) > 1
    ]
    shared_targets = set(shared_targets)

    for target in shared_targets:
        conflicting_sources = [
            source
            for source, swap_target in candidate_swaps.items()
            if swap_target == target
        ]

        target_node = fit_body.get_node(target)
        distances = {
            source: np.linalg.norm(
                active_body.get_node(source).position - target_node.position
            )
            for source in conflicting_sources
        }
        closest_source = min(distances, key=distances.get)
        for source in conflicting_sources:
            if source != closest_source:
                invalid_swaps.append(source)
                LOGGER.debug(
                    f"Removed swap for marker {source} (conflict with {closest_source})."
                )

    for source in invalid_swaps:
        del candidate_swaps[source]

    if set(candidate_swaps.values()) == set(candidate_swaps.keys()):
        # LOGGER.debug("All swaps corrected and are now valid.")
        return True

    # LOGGER.debug("Some swaps remain unresolved after correction.")
    return False


def generate_displacement_candidates(
    candidate_swaps: dict,
    fit_body: rb.RigidBody,
    active_body: rb.RigidBody,
    distance_tolerance: Union[str, float],
) -> list[str]:
    """!Generate a list of candidate displacements based on a radial tolerance.

    If a swap is staged to displace a marker (i.e. the marker has no corresponding swap in the active body),
    the swap is reconsidered. If the displacement marker is within the radial tolerance, the swapping marker is
    removed from the candidate swaps dictionary and considered for displacement.
    Otherwise, the displacement marker is added to the list of candidate displacements.

    @param candidate_swaps Dictionary of candidate swaps.
    @param fit_body Calibrated rigid body.
    @param active_body Active rigid body.
    @param distance_tolerance Radial tolerance for swapping.

    @return List of candidate displacements.
    """
    candidate_displacements = []
    displaced_sources = [
        k for k in candidate_swaps if k not in candidate_swaps.values()
    ]
    candidate_displacements = displaced_sources

    displaced_target_swaps = {
        k: v for k, v in candidate_swaps.items() if v not in candidate_swaps.keys()
    }
    for source, target in displaced_target_swaps.items():
        current = active_body.get_node(target)
        candidate = active_body.get_node(source)
        ref = fit_body.get_node(target)
        if current.exists:
            if np.linalg.norm(current.position - ref.position) > np.linalg.norm(
                candidate.position - ref.position
            ):
                candidate_displacements.append(target)
            else:
                del candidate_swaps[source]

    return candidate_displacements


def generate_swaps_and_displacements(
    calibrated_rigid_bodies: dict[str, rb.RigidBody],
    marker_trajectories: dict[str, MarkerTrajectory],
    distance_tolerance: float,
    trial_frames: list[int],
    frames: list[int],
) -> dict[str, list]:
    """!Generate swap and displacement candidates for each rigid body in the trial.

    @param calibrated_rigid_bodies Dictionary of calibrated rigid bodies.
    @param marker_trajectories Dictionary of marker trajectories.
    @param frames List of frames to process.
    @param distance_tolerance Radial tolerance for swapping.

    @return Dictionary of swap and displacement candidates for each rigid body, for each frame.
    """
    results = {}
    greedy_search_order = {rb_name: None for rb_name in calibrated_rigid_bodies}
    for rb_name, calib_body in calibrated_rigid_bodies.items():
        LOGGER.info(f"Processing rigid body: {rb_name}")
        if greedy_search_order[rb_name] is None:
            greedy_search_order[rb_name] = generate_greedy_search_order(calib_body)

        gso = greedy_search_order[rb_name]
        rb_results = []
        for iframe in tqdm.tqdm(frames, desc="Iterating through frames"):
            active_nodes = [
                traj.generate_node(m, iframe) for m, traj in marker_trajectories.items()
            ]
            active_body = copy.deepcopy(calib_body)
            active_body.update_node_positions(
                {n.marker: n.position for n in active_nodes},
                recompute_lengths=True,
                recompute_angles=True,
            )
            fit_body = copy.deepcopy(calib_body)
            rb.best_fit_transform(fit_body, active_body)

            candidate_swaps = generate_swap_candidates(
                fit_body, active_body, distance_tolerance, gso
            )

            candidate_displacements = []
            validated = validate_swap_candidates(candidate_swaps, fit_body, active_body)
            if not validated:
                candidate_displacements = generate_displacement_candidates(
                    candidate_swaps, fit_body, active_body, distance_tolerance
                )

            rb_results.append(
                {
                    "frame": trial_frames[iframe],
                    "swaps": candidate_swaps,
                    "displacements": candidate_displacements,
                }
            )

        results[rb_name] = rb_results

    return results


def plot_swaps(results: dict, start_frame: int = 0):
    """!Plot the number of swaps per frame for each rigid body in the trial.

    @param results Dictionary of swap and displacement candidates for each rigid body, for each frame.
    """
    ncols = len(results)
    nrows = 1

    fig, ax = plt.subplots(
        nrows, ncols, figsize=(5 * ncols, 4), sharex=True, sharey=True
    )

    for i, (rb_name, res) in enumerate(results.items()):
        num_swaps = [
            max(len(frame["swaps"]), len(frame["displacements"])) for frame in res
        ]
        frames = [frame["frame"] for frame in res]
        ax[i].plot(frames, num_swaps)
        ax[i].set_title(rb_name)
        ax[i].set_ylabel("Number of Swaps / Displacements")
        ax[i].set_xlabel("Frame Index")

    fig.tight_layout()

    return fig, ax


def swap_and_remove_markers_from_online_trial(
    vicon,
    subject_name: str,
    results: dict,
):
    """!Swap and remove markers from an online trial based on the results.

    @param vicon Vicon Nexus instance.
    @param subject_name Name of the subject.
    @param results Dictionary of swap and displacement candidates for each rigid body, for each frame.
    """
    for rb_name, res in results.items():
        for frame_res in tqdm.tqdm(
            res, desc=f"Swapping/removing markers from {rb_name}"
        ):
            vframe = frame_res["frame"]
            swaps = frame_res["swaps"]
            displacements = frame_res["displacements"]
            if not swaps and not displacements:
                continue

            relevant_markers = set([*swaps.keys(), *swaps.values(), *displacements])
            og_trajs = {}
            for marker in relevant_markers:
                x, y, z, e = vicon.GetTrajectoryAtFrame(subject_name, marker, vframe)
                og_trajs[marker] = (x, y, z, e)

            for swap_marker, target_marker in swaps.items():
                x, y, z, e = og_trajs[swap_marker]
                vicon.SetTrajectoryAtFrame(
                    subject_name, target_marker, vframe, x, y, z, e
                )

            for marker in displacements:
                x, y, z, _ = og_trajs[marker]
                vicon.SetTrajectoryAtFrame(subject_name, marker, vframe, x, y, z, False)

    LOGGER.info("Swapping and removing markers complete.")


def generate_string_summary(frame: int, swaps: dict, displacements: list) -> str:
    """!Generate a string summary of the swaps and displacements for a frame.

    @param frame Frame number.
    @param swaps Dictionary of swaps.
    @param displacements List of displacements.
    """
    swap_str = ", ".join([f"{k} <- {v}" for k, v in swaps.items() if v is not None])
    remove_str = ", ".join(displacements)
    return f"Frame: {frame} | Swapped: {swap_str or 'None'} | Removed: {remove_str or 'None'}"


def write_results_to_file(results: dict, output_fp: str, output_type: str = "txt"):
    """!Write the results to a file.

    @param results Dictionary of swap and displacement candidates for each rigid body, for each frame.
    @param output_fp Output file path.
    @param output_type Output file type (txt or json).
    """
    if output_type == "txt":
        with open(output_fp, "w") as f:
            for rb_name, res in results.items():
                f.write(f"Rigid Body: {rb_name}\n")
                for frame_res in res:
                    if frame_res["swaps"] or frame_res["displacements"]:
                        f.write(generate_string_summary(**frame_res) + "\n")

    elif output_type == "json":
        json_utils.export_dict_as_json(results, output_fp)
    else:
        LOGGER.error(f"Output type {output_type} is not supported.")


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
        help="Do not swap or remove markers from the trial.",
    )
    parser.add_argument(
        "-s",
        "--save_to_file",
        action="store_true",
        help="Saves the results to a file, i.e. a record of swaps and removals.",
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
        "--output_file_type",
        type=str,
        default="txt",
        choices=["json", "txt"],
        help="Output file type for swaps.",
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
        "--distance_tolerance",
        type=float,
        default=30,
        help="Marker distance tolerance from calibrated skeleton (in mm). Default is 30.",
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
        "--plot_swaps",
        action="store_true",
        help="Plot number of swaps per frame in the trial.",
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
    logger.set_root_logger(name="swap_rb_markers", mode=mode)

    if args.verbose:
        LOGGER.setLevel(logging.DEBUG)

    LOGGER.info("Running `swap_rb_markers.py` ...")
    LOGGER.debug(args)

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

    LOGGER.info(
        "Project: {}, Trial: {}, VSK: {}".format(
            *[os.path.basename(x) for x in [project_dir, trial_fp, vsk_fp]]
        ),
    )

    distance_tolerance = args.distance_tolerance

    ## Load Data
    calibrated_rigid_bodies = rigid_body_loader.get_rigid_bodies_from_vsk(vsk_fp)

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
    ## Process Data
    results = generate_swaps_and_displacements(
        calibrated_rigid_bodies,
        marker_trajectories,
        distance_tolerance,
        trial_frames,
        frames,
    )

    ## Plotting
    if args.plot_swaps:
        fig, ax = plot_swaps(results, start_frame=start)
        block = (offline or args.preserve_markers) and not args.inspect
        plt.show(block=block)

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
                trial_name = trial_fp.split(os.sep)[-1].split(".")[0]

                isa_args = [
                    "-pn",
                    project_dir,
                    "-sn",
                    subject_name,
                    "-tn",
                    trial_name,
                    "--frame",
                    str(ui),
                ]
                if offline:
                    isa_args.append("-off")
                if args.verbose:
                    isa_args.append("-v")

                LOGGER.info(
                    "Opening Interactive Score Analyzer as a subprocess. This may take a moement to load"
                )
                isa.run_as_subprocess(isa_args)

                LOGGER.info("ISA has shutdown.")

    if not (offline or args.preserve_markers):
        if args.force_true:
            ui = 0
        else:
            ui = hmi.get_user_input(
                "Swap and remove markers in trial? (y/n): ",
                exit_on_quit=True,
                choices=[hmi.YES_KEYWORDS, hmi.NO_KEYWORDS],
                num_tries=5,
            )
        if ui == 0:
            swap_and_remove_markers_from_online_trial(vicon, subject_name, results)

    ## Save Results
    if args.save_to_file:

        tn = trial_fp.split(os.sep)[-1].split(".")[0]
        output_fn = f"{tn}_swap_results"
        output_fp = directory.get_next_filename(
            project_dir, output_fn, args.output_file_type
        )
        write_results_to_file(results, output_fp, args.output_file_type)
        LOGGER.info(f"Results saved to: {output_fp}")

    LOGGER.info("Done.")


def test_main_with_args():
    sys.argv = [
        "swap_rb_markers.py",
        "-v",
        "-l",
        "-s",
        "-sn",
        "subject",
        "--start_frame",
        "7000",
        "--end_frame",
        "-1",
        "--plot_swaps",
        "--inspect",
    ]
    main()


if __name__ == "__main__":
    test_main_with_args()
    # main()
