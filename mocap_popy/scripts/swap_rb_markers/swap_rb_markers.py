"""!
    Swap RB Markers
    =================

    @author: C. McCarthy
"""

# %%
import os
import copy
import logging
from typing import Union

import numpy as np

import mocap_popy.models.rigid_body as rb
import mocap_popy.utils.vicon_quality_check as vqc
from mocap_popy.utils import rigid_body_loader

from viconnexusapi import ViconNexus

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
    radial_tolerance: Union[str, float],
    greedy_search_order: dict = None,
) -> dict:
    """!Generate a dictionary of candidate swaps based on a radial tolerance.

    If a marker is not within the radial tolerance of its corresponding marker in the fit body,
    the marker is considered a candidate for swapping.

    @param fit_body Calibrated rigid body, (after best fit to the active body).
    @param active_body Active rigid body.
    @param radial_tolerance Radial tolerance for swapping.

    @return Dictionary of candidate swaps.
    """
    greedy_search_order = greedy_search_order or generate_greedy_search_order(fit_body)
    candidate_swaps = {}
    for n in active_body.nodes:
        cn = fit_body.get_node(n.marker)
        if np.linalg.norm(n.position - cn.position) < radial_tolerance:
            continue
        for swap_marker in greedy_search_order[n.marker]:
            cn = fit_body.get_node(swap_marker)
            if np.linalg.norm(n.position - cn.position) < radial_tolerance:
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
    if set(candidate_swaps.values()) == set(candidate_swaps.keys()):
        LOGGER.info("All swaps are valid.")
        return True

    LOGGER.warning("Some swaps are invalid. Attempting to correct.")

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
        LOGGER.warning("Some swaps remain unresolved after correction.")
        return False


# %%
vicon = ViconNexus.ViconNexus()
# %%
subject_name = "subject"
radial_tolerance = 20  # mm

# %%
marker_trajectories = vqc.get_marker_trajectories(vicon, subject_name)

# %%

project_dir, trial_name = vicon.GetTrialName()

# %%

vsk_fp = os.path.join(os.path.normpath(project_dir), f"{subject_name}.vsk")
# %%

calibrated_bodies = rigid_body_loader.get_rigid_bodies_from_vsk(vsk_fp)

# %%
rb_name = "Right_Foot"
calibrated_body = calibrated_bodies[rb_name]

# %%

greedy_search_order = generate_greedy_search_order(calibrated_body)

# %%
frame = 8277
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

calibrated_body
# %%
active_body

# %%
swap_summaries = []
# %%


# %%
# candidate_swaps["Right_ML"] = "Right_MM"
candidate_swaps = generate_swap_candidates(
    calibrated_body, active_body, radial_tolerance, greedy_search_order
)

# %%
validate_swap_candidates(candidate_swaps, calibrated_body, active_body)

# %%

swap_str = ", ".join(
    [f"{k} -> {v}" for k, v in candidate_swaps.items() if v is not None]
)
remove_str = ", ".join([k for k, v in candidate_swaps.items() if v is None])
summary = (
    f"Frame: {frame} | Swapped: {swap_str or 'None'} | Removed: {remove_str or 'None'}"
)

# %%
candidate_swaps

# %%
summary
# %%
