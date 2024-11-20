"""!
    Rigid Body Scorer
    =================

    This module contains functions that help determine marker quality in a rigid body setup.

    @author C. McCarthy
"""

import copy
from typing import Optional, Literal, Union, Any
import logging

import numpy as np
import tqdm

from mocap_popy.models.rigid_body import RigidBody
from mocap_popy.models.marker_trajectory import MarkerTrajectory
from mocap_popy.scripts.unassign_rb_markers.scoring.scoringParameters import (
    ScoringParameters,
)

LOGGER = logging.getLogger("RigidBodyScorer")


def generate_residual_histories(
    initial_body: RigidBody,
    marker_trajectories: dict[str, MarkerTrajectory],
    residual_type: Literal["calib", "prior"],
    segments_only: bool = False,
    start_frame: bool = 0,
    end_frame: bool = -1,
    agg_params: Optional[ScoringParameters] = None,
):
    """!Generates histories of residuals for this body relative.

    @param initial_state Initial state (or calibration) of the rigid body.
    @param marker_trajectories Dictionary of marker trajectories.
    @param residual_type Type of residual to compute. Either "calib" or "prior".
    @param segments_only Whether to compute segment residuals only. Saves computation time.
    @param Start frame for residual computation. This corresponds to an index of the marker trajectories.
    @param End frame for residual computation. This corresponds to an index of the marker trajectories.
    @param agg_kwargs Keyword arguments for aggregate_residuals. i.e. average, ignore_tolerance.
                Can be specified for "segment" and "joint" components, or as a default for all components.
    """

    components = ["segment"]
    if not segments_only:
        components.append("joint")

    agg_params = ScoringParameters() if agg_params is None else agg_params
    LOGGER.debug(f"Aggregation parameters: {agg_params}")

    res_types = [f"residual_{residual_type}"]
    rb_trajs: dict[str, MarkerTrajectory] = {
        m: traj
        for m, traj in marker_trajectories.items()
        if m in initial_body.get_markers()
    }

    end = len(list(rb_trajs.values())[0].x) if end_frame == -1 else end_frame + 1
    total_iters = (end - start_frame) * len(initial_body.nodes)
    pbar = tqdm.tqdm(
        total=total_iters, desc=f"Generating residuals for {initial_body.name}: "
    )

    current_rigid_body = None
    prior_rigid_body = None
    ref_lengths = None
    if residual_type == "prior":
        ref_lengths = {str(seg): seg.length for seg in initial_body.segments}

    segment_residuals = []
    joint_residuals = None if segments_only else []

    for frame in range(start_frame, end):
        nodes = [traj.generate_node(m, frame) for m, traj in rb_trajs.items()]
        if residual_type == "calib" or current_rigid_body is None:
            calib_rigid_body = initial_body
            current_rigid_body = copy.deepcopy(initial_body)
        else:
            calib_rigid_body = None
            prior_rigid_body = copy.deepcopy(current_rigid_body)

        current_rigid_body.update_node_positions(
            nodes, recompute_lengths=True, recompute_angles=(not segments_only)
        )
        current_rigid_body.compute_segment_residuals(
            calib_rigid_body, prior_rigid_body, ref_lengths
        )
        if not segments_only:
            current_rigid_body.compute_joint_residuals(
                calib_rigid_body, prior_rigid_body
            )

        segments_res = []
        joint_res = []
        for node in current_rigid_body.nodes:
            pbar.update(1)
            avg = agg_params.aggregation_method.segment == "mean"
            ignore = not agg_params.preagg_thresholding.segment
            seg_res = current_rigid_body.get_aggregate_segment_residuals(
                node, res_types, avg, ignore
            )
            segments_res.append(seg_res)
            if segments_only:
                continue

            avg = agg_params.aggregation_method.joint == "mean"
            ignore = not agg_params.preagg_thresholding.joint
            j_res = current_rigid_body.get_aggregate_joint_residuals(
                node, res_types, avg, ignore
            )
            joint_res.append(j_res)

        segment_residuals.append(segments_res)
        if joint_residuals is not None:
            joint_residuals.append(joint_res)

    residual_histories = {"segment": segment_residuals, "joint": joint_residuals}
    return residual_histories


def score_residual(residual: float, tolerance: float = 0, weight: float = 1):
    """!Compute a score based on a residual, tolerance, and weight.

    Scores are positive values that are zero when the residual is less than the tolerance.

    @param residual Residual value.
    @param tolerance Tolerance value.
    @param weight Weight value.

    """
    return max(0, abs(residual) - tolerance) * weight


def score_rigid_body_components(
    rigid_body: RigidBody,
    component: Literal["segment", "joint", "node"],
    residual_type: Literal["calib", "prior"],
    scoring_parameters: ScoringParameters,
    marker_set: list[str] = None,
    return_composite_score: bool = False,
) -> dict[str, Union[float, list]]:
    """!Scores markers in a rigid body based on their segment residuals.

    @param component Component to score. Accepts "segment", "joint", or "node".

    @param rigid_body Rigid body to score.
    @param rigid_body_prior Rigid body with prior information (i.e. last frame, calibration).
    @param score_tolerance Tolerance for scoring (const for each marker unless specified as dict).
    @param score_weight Weight for scoring (const for each marker).
    @param aggregate_using_average Whether to aggregate residuals using the average.
    @param ignore_residual_tolerances Whether to ignore tolerances set at the segment/joint level.
    @param marker_set Set of markers to score.
    @param return_composite_score Applies to "node" component only. Whether to return a composite score.
            Otherwise, score for each marker includes list of individual scores, i.e. [composite, segment, joint].

    @param scores Dictionary of scores for each marker.
    """

    if component not in ["segment", "joint", "node"]:
        raise ValueError(f"Invalid component: {component}")

    if marker_set is None:
        marker_set = set([n.marker for n in rigid_body.nodes])

    if component == "node":
        sub_scores = {}
        for comp in ["segment", "joint"]:
            res = score_rigid_body_components(
                rigid_body,
                comp,
                residual_type,
                scoring_parameters=scoring_parameters,
                marker_set=marker_set,
            )
            w = getattr(scoring_parameters.aggregation_weight, comp)
            res = {
                m: r * w if isinstance(r, float) else [rr * w for rr in r]
                for m, r in res.items()
            }
            sub_scores[comp] = res
        combined = np.sum(
            [np.array(list(ss.values())) for ss in sub_scores.values()], axis=0
        )
        if return_composite_score:
            return {m: combined[i] for i, m in enumerate(marker_set)}

        return {
            m: [combined[i], *[ss[m] for ss in sub_scores.values()]]
            for i, m in enumerate(marker_set)
        }

    scores = {m: 0 for m in marker_set}
    average = getattr(scoring_parameters.aggregation_method, component) == "mean"
    ignore_residual_tolerances = not getattr(
        scoring_parameters.preagg_thresholding, component
    )

    args = [[f"residual_{residual_type}"], average, ignore_residual_tolerances]

    for m in marker_set:
        if component == "segment":
            res = rigid_body.get_aggregate_segment_residuals(m, *args)
        else:
            res = rigid_body.get_aggregate_joint_residuals(m, *args)
        scores[m] = res

    return scores


def sort_marker_scores(
    marker_scores: dict[str, float],
    threshold: Optional[Union[dict[str, float], float]] = None,
    max_markers: Optional[int] = None,
    include_duplicates: bool = False,
) -> list[tuple[str, float]]:
    """
    Sort scores and filter by threshold and maximum number of markers.

    @param marker_scores Dictionary of marker scores {marker_name: score}.
    @param threshold Minimum score to consider.
    @param max_markers Maximum number of markers to return.
    @param include_duplicates Whether to include markers with the same score (can exceed max_markers).
    """
    sorted_scores = sorted(marker_scores.items(), key=lambda x: x[1], reverse=True)

    if threshold is not None:
        if isinstance(threshold, float):
            sorted_scores = [
                marker for marker in sorted_scores if marker[1] > threshold
            ]
        else:
            new_scores = []
            for m in sorted_scores:
                if m[0] not in threshold:
                    LOGGER.debug(
                        f"Threshold not found for marker {m[0]}. Skipping marker."
                    )
                    continue
                if m[1] > threshold[m[0]]:
                    new_scores.append(m)

            sorted_scores = new_scores

    if max_markers is not None and len(sorted_scores) > max_markers:
        last_candidate = sorted_scores[max_markers - 1][1]
        excluded_duplicates = [
            marker
            for marker in sorted_scores[max_markers:]
            if marker[1] == last_candidate
        ]
        if include_duplicates:
            sorted_scores = sorted_scores[:max_markers] + excluded_duplicates
        elif len(excluded_duplicates) > 0:
            sorted_scores = [
                marker for marker in sorted_scores if marker[1] > last_candidate
            ]

        sorted_scores = sorted_scores[:max_markers]

    return sorted_scores


def sort_non_composite_marker_scores(
    marker_scores: dict[str, list],
    sort_order: list[str] = None,
    threshold: Optional[float] = None,
    max_markers: Optional[int] = None,
) -> list[tuple[str, list]]:
    """
    Sort marker scores that are represented by lists of [composite, segment, joint] scores.
    Additionally filters by threshold (based on first element in list) and maximum number of markers.

    @param marker_scores Dictionary of marker scores {marker_name: [composite, segment, joint]}.
    @param sort_order Order to sort markers by. Default is ["composite", "segment", "joint"].
    @param threshold Minimum score to consider.
    @param max_markers Maximum number of markers to return.
    """

    components = ["composite", "segment", "joint"]
    order = [0, 1, 2]
    if sort_order is not None:
        order = [components.index(comp) for comp in sort_order]

    sorted_scores = sorted(
        marker_scores.items(),
        key=lambda x: tuple([x[1][i] for i in order]),
        reverse=True,
    )

    if threshold is not None:
        sorted_scores = [score for score in sorted_scores if score[1][0] > threshold]

    if max_markers is not None:
        sorted_scores = sorted_scores[:max_markers]

    return sorted_scores
