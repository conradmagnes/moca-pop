"""!
    Interactive score analyzer helper functions.
    ============================================

    @author C. McCarthy
"""

import numpy as np
import plotly.graph_objs as go


from mocap_popy.config import regex
from mocap_popy.models.rigid_body import RigidBody
from mocap_popy.scripts.unassign_rb_markers.scoring import scoringParameters

import mocap_popy.aux_scripts.interactive_score_analyzer.constants as isa_consts


def refactor_removal_threshold(
    scoring_params: scoringParameters.ScoringParameters, marker_names: list[str]
):
    """!Refactor the removal threshold to be a dictionary with marker names as keys.

    @param scoring_params Scoring parameters object.
    @param marker_names List of marker names.
    """
    if isinstance(scoring_params.removal_threshold, dict):
        removal_threshold_all = 0
        for m in marker_names:
            if m not in scoring_params.removal_threshold:
                scoring_params.removal_threshold[m] = 0
    else:
        removal_threshold_all = scoring_params.removal_threshold
        scoring_params.removal_threshold = {
            m: scoring_params.removal_threshold for m in marker_names
        }

    return scoring_params, removal_threshold_all


def remove_symmetry_from_scoring(scoring_params: scoringParameters.ScoringParameters):
    """!Remove symmetry from the scoring parameters.

    @param scoring_params Scoring parameters object.
    """
    removal_threshold = {}
    for k, v in scoring_params.removal_threshold.items():
        try:
            _, symm_comp = regex.parse_symmetrical_component(k)
        except ValueError:
            symm_comp = k
        removal_threshold[symm_comp] = v

    scoring_params.removal_threshold = removal_threshold
    return scoring_params


def generate_node_trace(
    body: RigidBody,
    node_scores: dict[str, float],
    mode: str,
    marker_args: dict,
    text_position="top center",
    body_type="calibrated",
):
    exisiting_nodes = [n for n in body.nodes if n.exists]
    exisiting_node_scores = [node_scores[n.marker] for n in exisiting_nodes]
    if body_type == "calibrated":
        margs = {**marker_args, "color": isa_consts.CALIBRATED_COLOR}
    else:
        colors = [isa_consts.get_component_color(s) for s in exisiting_node_scores]
        margs = {**marker_args, "color": colors}

    return go.Scatter3d(
        x=[n.position[0] for n in exisiting_nodes],
        y=[n.position[1] for n in exisiting_nodes],
        z=[n.position[2] for n in exisiting_nodes],
        mode=mode,
        marker=margs,
        text=[n.marker for n in exisiting_nodes],
        textposition=text_position,
    )


def generate_segment_traces(body: RigidBody, line_args: dict, body_type="calibrated"):
    traces = []
    for seg in [s for s in body.segments if s.exists]:
        if body_type == "calibrated":
            color = isa_consts.CALIBRATED_COLOR
        else:
            color = isa_consts.get_component_color(
                abs(seg.residual_calib), seg.tolerance
            )

        t = go.Scatter3d(
            x=[seg.nodes[0].position[0], seg.nodes[1].position[0]],
            y=[seg.nodes[0].position[1], seg.nodes[1].position[1]],
            z=[seg.nodes[0].position[2], seg.nodes[1].position[2]],
            mode="lines",
            line={**line_args, "color": color},
            hoverinfo="skip",
            hovertemplate=None,
        )
        traces.append(t)

    return traces
