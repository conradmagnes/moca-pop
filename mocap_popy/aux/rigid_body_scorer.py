"""!
    Rigid Body Scorer
    =================

    This module contains functions that help determine marker quality in a rigid body setup.

    @author C. McCarthy
"""

from typing import Optional, Literal, Union, Any

import numpy as np

from mocap_popy.models.rigid_body import RigidBody


def score_residual(residual: float, tolerance: float = 0, weight: float = 1):
    """!Compute a score based on a residual, tolerance, and weight.

    Scores are positive values that are zero when the residual is less than the tolerance.

    @param residual Residual value.
    @param tolerance Tolerance value.
    @param weight Weight value.

    """
    return max(0, abs(residual) - tolerance) * weight


def score_rigid_body_components(
    component: Literal["segment", "joint", "node"],
    rigid_body: RigidBody,
    rigid_body_prior: RigidBody,
    score_tolerance: Union[float, dict] = 0,
    score_weight: Union[float, dict] = 0,
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

    def get_dict_value(key: list, value: Union[dict, Any]):
        if isinstance(value, dict):
            return value.get(key, 0)
        return value

    if component not in ["segment", "joint", "node"]:
        raise ValueError(f"Invalid component: {component}")

    if marker_set is None:
        marker_set = [n.marker for n in rigid_body.nodes]
        marker_set.extend([n.marker for n in rigid_body_prior.nodes])
        marker_set = set(marker_set)

    if component == "node":
        sub_scores = {}
        for comp in ["segment", "joint"]:
            res = score_rigid_body_components(
                comp,
                rigid_body,
                rigid_body_prior,
                score_tolerance=get_dict_value(comp, score_tolerance),
                score_weight=get_dict_value(comp, score_weight),
                aggregate_using_average=get_dict_value(comp, aggregate_using_average),
                ignore_residual_tolerance=get_dict_value(
                    comp, ignore_residual_tolerances
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
                rigid_body_prior,
                aggregate_using_average,
                ignore_residual_tolerances,
            )
        else:
            res = rigid_body.get_aggregate_joint_residual(
                m,
                rigid_body_prior,
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
