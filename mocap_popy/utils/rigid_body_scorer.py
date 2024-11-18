"""!
    Rigid Body Scorer
    =================

    This module contains functions that help determine marker quality in a rigid body setup.

    @author C. McCarthy
"""

import copy
from typing import Optional, Literal, Union, Any


import numpy as np

from mocap_popy.models.rigid_body import RigidBody
from mocap_popy.models.marker_trajectory import MarkerTrajectory


def validate_segment_and_joint_kwargs(
    kwargs: dict, defaults: dict, components: list[str] = None
):
    """!Validates keyword arguments for segment and joint components.

    @param kwargs Keyword arguments.
    @param defaults Default keyword arguments.
    @param components List of components to validate. Default is ["segment", "joint"].

    @return Validated keyword arguments.
    """
    comps = ["segment", "joint"] if components is None else components
    valid_kwargs = {}
    for comp in comps:
        valid_kwargs[comp] = (
            copy.deepcopy(defaults)
            if comp not in defaults
            else copy.deepcopy(defaults[comp])
        )

    if kwargs is not None:
        for comp in comps:
            for k, v in valid_kwargs[comp].items():
                if comp in kwargs:
                    valid_kwargs[comp][k] = kwargs[comp].get(k, v)
                else:
                    valid_kwargs[comp][k] = kwargs.get(k, v)

    return valid_kwargs


def generate_residual_histories(
    initial_bodies: dict[str, RigidBody],
    marker_trajectories: dict[str, MarkerTrajectory],
    residual_type: Literal["calib", "prior"],
    segments_only: bool = False,
    start_frame: bool = 0,
    end_frame: bool = -1,
    agg_kwargs: dict = None,
):
    """!Generates residual histories for rigid bodies.

    @param bodies Dictionary of rigid bodies.
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

    default_agg = dict(average=True, ignore_tolerance=False)
    agg_k = validate_segment_and_joint_kwargs(agg_kwargs, default_agg, components)

    residual_histories = {}
    for rb_name, rb in initial_bodies.items():
        prior_rigid_body = None

        residual_histories[rb_name] = {comp: None for comp in components}
        rb_trajs: dict[str, MarkerTrajectory] = {
            m: traj for m, traj in marker_trajectories.items() if m in rb.get_markers()
        }

        end = len(list(rb_trajs.values())[0].x) if end_frame == -1 else end_frame + 1

        segment_residuals = []
        joint_residuals = []

        for frame in range(start_frame, end):
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

            segments_res = []
            joint_res = []
            for node in current_rigid_body.nodes:
                seg_res = current_rigid_body.get_aggregate_segment_residuals(
                    node, res_types=[f"residual_{residual_type}"], **agg_k["segment"]
                )
                segments_res.append(seg_res)
                if segments_only:
                    continue
                j_res = current_rigid_body.get_aggregate_joint_residuals(
                    node, res_types=[f"residual_{residual_type}"], **agg_k["joint"]
                )
                joint_res.append(j_res)

            segment_residuals.append(segments_res)
            if not segments_only:
                joint_residuals.append(joint_res)

        residual_histories[rb_name]["segment"] = segment_residuals
        if not segments_only:
            residual_histories[rb_name]["joint"] = joint_residuals

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
    score_tolerance: Union[float, dict] = 0,
    score_weight: Union[float, dict] = 1,
    aggregate_using_average: Union[bool, dict] = True,
    ignore_residual_tolerances: Union[bool, dict] = False,
    marker_set: list[str] = None,
    return_composite_score: bool = False,
) -> dict[str, Union[float, list]]:
    """!Scores markers in a rigid body based on their segment residuals.

    @param component Component to score. Accepts "segment", "joint", or "node".

    @param rigid_body Rigid body to score.
    @param rigid_body_prior Rigid body with prior information (i.e. last frame, calibration).
    @param score_tolerance Tolerance for scoring.
    @param score_weight Weight for scoring.
    @param aggregate_using_average Whether to aggregate residuals using the average.
    @param ignore_residual_tolerances Whether to ignore residual tolerances.
    @param marker_set Set of markers to score.
    @param return_composite_score Applies to "node" component only. Whether to return a composite score.
            Otherwise, score for each marker includes list of individual scores, i.e. [composite, segment, joint].

    @param scores Dictionary of scores for each marker.
    """

    def get_dict_value(key: list, value: Union[dict, Any], default: Any = 0):
        if isinstance(value, dict):
            return value.get(key, default)
        return value

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
                score_tolerance=get_dict_value(comp, score_tolerance),
                score_weight=get_dict_value(comp, score_weight),
                aggregate_using_average=get_dict_value(
                    comp, aggregate_using_average, True
                ),
                ignore_residual_tolerances=get_dict_value(
                    comp, ignore_residual_tolerances, False
                ),
                marker_set=marker_set,
            )
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
    for m in marker_set:
        if component == "segment":
            res = rigid_body.get_aggregate_segment_residuals(
                m,
                [f"residual_{residual_type}"],
                aggregate_using_average,
                ignore_residual_tolerances,
            )
        else:
            res = rigid_body.get_aggregate_joint_residuals(
                m,
                [f"residual_{residual_type}"],
                aggregate_using_average,
                ignore_residual_tolerances,
            )
        scores[m] = score_residual(res, score_tolerance, score_weight)

    return scores


def sort_marker_scores(
    marker_scores: dict[str, float],
    threshold: Optional[float] = None,
    max_markers: Optional[int] = None,
) -> list[tuple[str, float]]:
    """
    Sort scores and filter by threshold and maximum number of markers.

    @param marker_scores Dictionary of marker scores {marker_name: score}.
    @param threshold Minimum score to consider.
    @param max_markers Maximum number of markers to return.
    """
    sorted_scores = sorted(marker_scores.items(), key=lambda x: x[1], reverse=True)

    if threshold is not None:
        sorted_scores = [marker for marker in sorted_scores if marker[1] > threshold]

    if max_markers is not None:
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
