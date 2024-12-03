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

import numpy as np

import mocap_popy.config.logger as logger
import mocap_popy.models.rigid_body as rb
import mocap_popy.utils.quality_check as qc
from mocap_popy.utils import rigid_body_loader, argparse_utils, dist_utils, c3d_parser


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
    if len(candidate_swaps) < 1:
        return True

    if set(candidate_swaps.values()) == set(candidate_swaps.keys()):
        LOGGER.debug("All swaps are valid.")
        return True

    LOGGER.info("Some swaps are invalid. Attempting to correct.")

    invalid_swaps = []
    shared_targets = [
        target
        for target in candidate_swaps.values()
        if list(candidate_swaps.values()).count(target) > 1
    ]
    shared_targets = set(shared_targets)

    for target in shared_targets:
        conflicting_markers = [
            m for m, swap_target in candidate_swaps.items() if swap_target == target
        ]

        target_node = fit_body.get_node(target)
        distances = {
            m: np.linalg.norm(active_body.get_node(m).position - target_node.position)
            for m in conflicting_markers
        }
        closest_marker = min(distances, key=distances.get)
        for marker in conflicting_markers:
            if marker != closest_marker:
                candidate_swaps[marker] = None
                invalid_swaps.append(marker)
                LOGGER.debug(
                    f"Removed swap for marker {marker} (conflict with {closest_marker})."
                )

    for marker in invalid_swaps:
        del candidate_swaps[marker]

    if set(candidate_swaps.values()) == set(candidate_swaps.keys()):
        LOGGER.info("All swaps corrected and are now valid.")
        return True
    else:
        LOGGER.info("Some swaps remain unresolved after correction.")
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
    displacing_swaps = {
        k: v for k, v in candidate_swaps.items() if v not in candidate_swaps.keys()
    }
    for key, marker in displacing_swaps.items():
        an = active_body.get_node(marker)
        fn = fit_body.get_node(marker)
        if np.linalg.norm(an.position - fn.position) < distance_tolerance:
            candidate_displacements.append(key)
            del candidate_swaps[key]
        else:
            candidate_displacements.append(marker)

    return candidate_displacements


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
        markers = vicon.GetMarkerNames(subject_name)
        marker_trajectories = {}
        for m in markers:
            x, y, z, e = vicon.GetTrajectory(subject_name, m)
            marker_trajectories[m] = MarkerTrajectory(x, y, z, e)

        trial_range = vicon.GetTrialRange()
        trial_frames = list(range(trial_range[0], trial_range[1] + 1))

    start, end = argparse_utils.validate_start_end_frames(
        args.start_frame, args.end_frame, trial_frames
    )
    frames = list(range(start - trial_frames[0], end - trial_frames[0] + 1))


# %%
vicon = ViconNexus.ViconNexus()
# %%
subject_name = "subject"
distance_tolerance = 30  # mm

# %%
marker_trajectories = qc.get_marker_trajectories(vicon, subject_name)

# %%

project_dir, trial_name = vicon.GetTrialName()

# %%

vsk_fp = os.path.join(os.path.normpath(project_dir), f"{subject_name}.vsk")
calibrated_bodies = rigid_body_loader.get_rigid_bodies_from_vsk(vsk_fp)

# %%
rb_name = "Right_Foot"
calibrated_body = calibrated_bodies[rb_name]

# %%

greedy_search_order = generate_greedy_search_order(calibrated_body)

# %%
frame = 7792
active_nodes = [traj.generate_node(m, frame) for m, traj in marker_trajectories.items()]
active_body = copy.deepcopy(calibrated_body)
active_body.update_node_positions(
    {n.marker: n.position for n in active_nodes},
    recompute_lengths=True,
    recompute_angles=True,
)
# %%

active_body
# %%
rb.best_fit_transform(calibrated_body, active_body)
# %%

# %%
swap_summaries = []
# %%


# %%
candidate_swaps = generate_swap_candidates(
    calibrated_body, active_body, distance_tolerance, greedy_search_order
)
candidate_swaps
# %%

candidate_displacements = []
validated = validate_swap_candidates(candidate_swaps, calibrated_body, active_body)
if not validated:
    candidate_displacements = generate_displacement_candidates(
        candidate_swaps, calibrated_body, active_body, distance_tolerance
    )

candidate_displacements

# %%

swap_str = ", ".join(
    [f"{k} -> {v}" for k, v in candidate_swaps.items() if v is not None]
)
remove_str = ", ".join(candidate_displacements)
summary = (
    f"Frame: {frame} | Swapped: {swap_str or 'None'} | Removed: {remove_str or 'None'}"
)
summary

# %%
candidate_swaps

# %%
summary
# %%
