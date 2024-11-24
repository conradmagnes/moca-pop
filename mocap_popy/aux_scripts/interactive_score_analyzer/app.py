"""

    Interactive Web-based Application to Test Scoring Strategies for Rigid Body Calibration
    =======================================================================================

    This script should user `dash` to create an interactive web-based application that allows
    users to move nodes in 3D space and see the connected segments and joints for each node.
    
    TODO:
    - remove / add markers controls
    -   handle non-existent markers from file
    - scaling of x axis is off?

"""

import argparse
import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import copy
import os
import json
import logging
import tkinter as tk
from tkinter import filedialog
import base64

import numpy as np
import plotly.graph_objs as go
import pydantic

import mocap_popy.config.directory as directory
import mocap_popy.config.regex as regex
from mocap_popy.models.rigid_body import Node, Segment, Joint, RigidBody
from mocap_popy.utils import rigid_body_loader, json_utils, c3d_parser
from mocap_popy.scripts.unassign_rb_markers.scoring import scorer, scoringParameters

import mocap_popy.aux_scripts.interactive_score_analyzer.constants as isa_consts
import mocap_popy.aux_scripts.interactive_score_analyzer.layout as isa_layout

LOGGER = logging.getLogger("InteractiveScoreAnalyzer")
## ---- ##


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


def best_fit_transform(target_body: RigidBody, active_body: RigidBody):
    """
    Computes the optimal rigid transformation (rotation + translation) that aligns
    points A with points B using the Kabsch algorithm.

    @param target_body Rigid body to be transformed.
    @param active_body Rigid body to be used as the reference.
    """
    active_nodes = [n for n in active_body.nodes if n.exists]
    active_markers = [n.marker for n in active_nodes]
    active_positions = np.array([n.position for n in active_nodes])
    calibrated_positions = np.array(
        [target_body.get_node(m).position for m in active_markers]
    )

    A = calibrated_positions
    B = active_positions

    # 1. Calculate centroids of A and B
    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)

    # 2. Translate points to have their centroids at the origin
    AA = A - centroid_A
    BB = B - centroid_B

    # 3. Compute the covariance matrix H
    H = AA.T @ BB

    # 4. Compute the Singular Value Decomposition (SVD) of H
    U, S, Vt = np.linalg.svd(H)

    # 5. Compute the optimal rotation matrix R
    R = Vt.T @ U.T

    # Special reflection case to correct improper rotation matrices
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T

    # 6. Compute the optimal translation vector t
    t = centroid_B - R @ centroid_A

    all_calibrated_positions = np.array([n.position for n in target_body.nodes])
    transformed_points = apply_transform(all_calibrated_positions, R, t)
    target_body.update_node_positions(
        {m: p for m, p in zip(target_body.get_markers(), transformed_points)}
    )

    return


def apply_transform(points, R, t):
    """
    Applies the given rigid transformation (rotation + translation) to the points.

    @param points (N, 3) ndarray of points to transform.
    @param R (3, 3) ndarray representing the rotation matrix.
    @param t (3, ) ndarray representing the translation vector.

    @return transformed_points (N, 3) ndarray of transformed points.
    """
    return (R @ points.T).T + t


project_dir = os.path.join(directory.DATASET_DIR, "shoe_stepping")
vsk_fp = os.path.join(project_dir, "subject.vsk")

ignore_symmetry = True
calibrated_rigid_bodies = rigid_body_loader.get_rigid_bodies_from_vsk(
    vsk_fp, ignore_marker_symmetry=ignore_symmetry
)

rb_name = "Right_Foot"

calibrated_body = calibrated_rigid_bodies[rb_name]
calibrated_body.compute_segment_lengths()
calibrated_body.compute_joint_angles()


trial_fp = os.path.join(project_dir, "rom_calibration.c3d")
c3d_reader = c3d_parser.get_reader(trial_fp)
marker_trajectories = c3d_parser.get_marker_trajectories(c3d_reader)
trial_frames = c3d_parser.get_frames(c3d_reader)

if ignore_symmetry:
    rb_side, _ = regex.parse_symmetrical_component(rb_name)
    sym_trajectories = {}
    for m, t in marker_trajectories.items():
        try:
            symm_side, symm_comp = regex.parse_symmetrical_component(m)
            if symm_side == rb_side:
                sym_trajectories[symm_comp] = t
        except ValueError:
            continue
    marker_trajectories = sym_trajectories

active_frame = 0
active_nodes = [
    traj.generate_node(m, active_frame) for m, traj in marker_trajectories.items()
]
active_body = copy.deepcopy(calibrated_body)
active_body.update_node_positions({n.marker: n.position for n in active_nodes})

best_fit_transform(calibrated_body, active_body)

scoring_params = scoringParameters.ScoringParameters()
scoring_params, removal_threshold_all = refactor_removal_threshold(
    scoring_params, calibrated_body.get_markers()
)
if ignore_symmetry:
    scoring_params = remove_symmetry_from_scoring(scoring_params)

# Initial Node Positions
initial_positions = {m.marker: m.position for m in active_body.nodes}

component_scores = {
    comp_name: {m.marker: 0 for m in calibrated_body.nodes}
    for comp_name in ["segment", "joint"]
}
raw_node_scores = {m: 0 for m in active_body.get_markers()}
node_scores = {m: 0 for m in active_body.get_markers()}


## ------ Graphing Helpers ----- ##


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


# Initialize the Dash app
app = dash.Dash(__name__, suppress_callback_exceptions=True)

stores = [
    dcc.Store(id="selected-node", data=active_body.get_markers()[0]),
    dcc.Store(id="node-moved", data=False),
    dcc.Store(id="node-positions", data=initial_positions),
    dcc.Store(id="node-offsets", data={node: [0, 0, 0] for node in initial_positions}),
    dcc.Store(id="component-scores", data=component_scores),
    dcc.Store(id="raw-node-scores", data=raw_node_scores),
    dcc.Store(id="node-scores", data=node_scores),
    dcc.Store(id="max-score", data=0.0),
    dcc.Store(id="max-marker", data="N/A"),
    dcc.Store(id="scoring-params", data=scoring_params.model_dump_json()),
    dcc.Store(id="removal-threshold-all", data=removal_threshold_all),
]

# Layout of the App
app.layout = html.Div(
    [
        isa_layout.generate_ni_div(),
        isa_layout.generate_ig(),
        isa_layout.generate_nm_div(),
        isa_layout.generate_spe_div(scoring_params, removal_threshold_all),
        *stores,
    ]
)


# Callback to update the 3D plot
@app.callback(
    Output("interactive-graph", "figure"),
    [Input("node-positions", "data"), Input("node-scores", "data")],
)
def update_graph(node_positions, node_scores):
    calib_node_trace = generate_node_trace(
        calibrated_body,
        node_scores,
        "markers",
        dict(size=5),
    )
    active_node_trace = generate_node_trace(
        active_body,
        node_scores,
        "markers+text",
        dict(size=7),
        body_type="active",
    )

    calib_segment_traces = generate_segment_traces(calibrated_body, dict(width=1))
    active_segment_traces = generate_segment_traces(
        active_body, dict(width=2), body_type="active"
    )
    data = (
        calib_segment_traces
        + [calib_node_trace]
        + active_segment_traces
        + [active_node_trace]
    )

    existing_pos = [n.position for n in active_body.nodes if n.exists]
    existing_pos += [n.position for n in calibrated_body.nodes if n.exists]

    scene_limits = isa_layout.get_scene_limits(existing_pos, isa_layout.NM_SLIDER_ARGS)

    layout = go.Layout(
        margin=dict(l=0, r=0, b=0, t=0),
        scene={"aspectmode": "cube", **scene_limits},
        uirevision="same",
        showlegend=False,
    )

    return go.Figure(data=data, layout=layout)


# Callback to update which node is selected based on click
@app.callback(
    Output("selected-node", "data"),
    Input("interactive-graph", "clickData"),
)
def select_node(click_data):
    if click_data is None:
        return active_body.get_markers()[0]

    node_name = click_data["points"][0]["text"]
    return node_name


@app.callback(
    Output("node-offsets", "data"),
    [
        Input("slider-x", "value"),
        Input("slider-y", "value"),
        Input("slider-z", "value"),
        Input("reset-all-button", "n_clicks"),
    ],
    [State("selected-node", "data"), State("node-offsets", "data")],
)
def update_offsets(
    slider_x, slider_y, slider_z, reset_all_clicks, selected_node, node_offsets
):
    ctx = dash.callback_context
    if ctx.triggered and ctx.triggered[0]["prop_id"] == "reset-all-button.n_clicks":
        return {node: [0, 0, 0] for node in node_offsets}
    node_offsets[selected_node] = [slider_x, slider_y, slider_z]
    return node_offsets


# Callback to update the sliders when a node is selected or when reset buttons are clicked
@app.callback(
    [
        Output("slider-x", "value"),
        Output("slider-y", "value"),
        Output("slider-z", "value"),
        Output("slider-label", "children"),
    ],
    [
        Input("selected-node", "data"),
        Input("reset-button", "n_clicks"),
        Input("reset-all-button", "n_clicks"),
    ],
    State("node-offsets", "data"),
)
def update_sliders(selected_node, reset_clicks, reset_all_clicks, node_offsets):
    ctx = dash.callback_context
    if ctx.triggered:
        triggered_button = ctx.triggered[0]["prop_id"].split(".")[0]
        if triggered_button in ["reset-button", "reset-all-button"]:
            return (0, 0, 0, dash.no_update)

    return (
        *node_offsets[selected_node],
        f"{isa_layout.NM_HEADER_LABEL}: {selected_node}",
    )


# Combined Callback to update node positions, offsets, and handle "Reset All" functionality
@app.callback(
    [Output("node-positions", "data"), Output("node-moved", "data")],
    [
        Input("node-offsets", "data"),
        Input("reset-all-button", "n_clicks"),
    ],
    [
        State("selected-node", "data"),
        State("node-positions", "data"),
    ],
)
def update_node_positions(
    node_offsets,
    reset_all_clicks,
    selected_node,
    node_positions,
):
    ctx = dash.callback_context

    updated_positions = node_positions.copy()

    if ctx.triggered and ctx.triggered[0]["prop_id"] == "reset-all-button.n_clicks":
        # Reset all node positions and offsets
        updated_positions = {
            node: initial_positions[node]
            for node in initial_positions
            if active_body.get_node(node).exists
        }
        active_body.update_node_positions(
            updated_positions, recompute_lengths=True, recompute_angles=True
        )
        active_body.compute_segment_residuals(calibrated_body)
        active_body.compute_joint_residuals(calibrated_body)

    else:
        offsets = node_offsets[selected_node]

        initial_position = initial_positions[selected_node]
        updated_positions[selected_node] = [
            ip + off for ip, off in zip(initial_position, offsets)
        ]

        if updated_positions[selected_node] == node_positions[selected_node]:
            return node_positions, False

        active_node = active_body.get_node(selected_node)
        active_node.position = np.array(updated_positions[selected_node])
        for seg in active_node.get_connected_segments(active_body.segments):
            seg.compute_length()
            seg.compute_residuals(calibrated_body)
        for joint in active_node.get_connected_joints(active_body.joints):
            joint.compute_angle()
            joint.compute_residuals(calibrated_body)

    return updated_positions, True


@app.callback(
    [
        Output("component-scores", "data"),
        Output("raw-node-scores", "data"),
        Output("node-scores", "data"),
        Output("max-score", "data"),
        Output("max-marker", "data"),
    ],
    [
        Input("node-positions", "data"),
        Input("update-button", "n_clicks"),
        Input("scoring-params", "data"),
    ],
    [
        State("node-moved", "data"),
        State("scoring-params", "data"),
        State("component-scores", "data"),
        State("raw-node-scores", "data"),
        State("node-scores", "data"),
        State("max-score", "data"),
        State("max-marker", "data"),
    ],
)
def update_scores(
    node_positions,
    update_button_clicks,
    scoring_params_input,
    node_moved,
    scoring_params_state,
    component_scores,
    raw_node_scores,
    node_scores,
    max_score,
    max_marker,
):
    # if not node_moved and
    ctx = dash.callback_context
    updated_triggered = (
        ctx.triggered and ctx.triggered[0]["prop_id"] == "update-button.n_clicks"
    )
    if not (node_moved or updated_triggered):
        return component_scores, raw_node_scores, node_scores, max_score, max_marker
    scoring_params = scoringParameters.ScoringParameters.model_validate_json(
        scoring_params_state
    )

    updated_component_scores = {
        comp: scorer.score_rigid_body_components(
            active_body,
            component=comp,
            residual_type="calib",
            scoring_parameters=scoring_params,
        )
        for comp in ["segment", "joint"]
    }
    updated_raw_node_scores = scorer.combine_component_scores(
        updated_component_scores, scoring_params, return_composite_score=True
    )
    updated_node_scores = {
        m: max(0, rs - scoring_params.removal_threshold.get(m, 0))
        for m, rs in updated_raw_node_scores.items()
    }
    updated_max_score = max(updated_node_scores.values())
    updated_max_marker = [
        m for m, v in updated_node_scores.items() if v == updated_max_score
    ][0]

    if updated_max_score == 0:
        updated_max_marker = "N/A"

    return (
        updated_component_scores,
        updated_raw_node_scores,
        updated_node_scores,
        updated_max_score,
        updated_max_marker,
    )


# Callback to display connected segments and joints
@app.callback(
    Output("node-inspector", "children"),
    [
        Input("selected-node", "data"),
        Input("node-positions", "data"),
        Input("node-offsets", "data"),
        Input("component-scores", "data"),
        Input("raw-node-scores", "data"),
        Input("node-scores", "data"),
        Input("max-score", "data"),
        Input("max-marker", "data"),
        Input("scoring-params", "data"),
    ],
    State("scoring-params", "data"),
)
def update_node_inspector(
    selected_node,
    node_positions,
    node_offsets,
    component_scores,
    raw_node_scores,
    node_scores,
    max_score,
    max_marker,
    scoring_params_input,
    scoring_params_state,
):

    scoring_params = scoringParameters.ScoringParameters.model_validate_json(
        scoring_params_state
    )
    connected_segments = active_body.get_connected_segments(selected_node)
    connected_joints = active_body.get_connected_joints(selected_node)

    return isa_layout.generate_ni_div_children(
        selected_node,
        connected_segments,
        connected_joints,
        component_scores,
        raw_node_scores,
        node_scores,
        max_score,
        max_marker,
        scoring_params,
    )


@app.callback(
    Output("removal-threshold-label", "children"),
    Input("removal-threshold-type", "value"),
)
def update_removal_threshold_label(removal_threshold_type: str):
    return f"Threshold for {removal_threshold_type}:"


# Callback to update scoring parameters based on input fields
@app.callback(
    [Output("scoring-params", "data"), Output("removal-threshold-all", "data")],
    [Input("update-button", "n_clicks"), Input("import-scoring", "contents")],
    [
        State("import-scoring", "filename"),
        State("preagg-segment", "value"),
        State("preagg-joint", "value"),
        State("agg-method-segment", "value"),
        State("agg-method-joint", "value"),
        State("agg-weight-segment", "value"),
        State("agg-weight-joint", "value"),
        State("removal-threshold-type", "value"),
        State("removal-threshold-value", "value"),
        State("scoring-params", "data"),
        State("removal-threshold-all", "data"),
    ],
)
def update_scoring_parameters(
    update_button_clicks,
    import_contents,
    import_filename,
    preagg_segment,
    preagg_joint,
    agg_method_segment,
    agg_method_joint,
    agg_weight_segment,
    agg_weight_joint,
    removal_threshold_type,
    removal_threshold_value,
    scoring_params_data,
    removal_threshold_all,
):

    ctx = dash.callback_context
    if ctx.triggered and ctx.triggered[0]["prop_id"] == "update-button.n_clicks":
        scoring_data = update_scoring_params_from_dash(
            preagg_segment,
            preagg_joint,
            agg_method_segment,
            agg_method_joint,
            agg_weight_segment,
            agg_weight_joint,
            removal_threshold_type,
            removal_threshold_value,
            scoring_params_data,
        )
        updated_threshold_all = removal_threshold_all
    else:
        if import_contents is None or not import_filename.endswith(".json"):
            raise dash.exceptions.PreventUpdate

        try:
            scoring_data, updated_threshold_all = update_scoring_params_from_file(
                import_contents
            )
        except ValueError as e:
            raise dash.exceptions.PreventUpdate

    LOGGER.debug(f"Updated Scoring Parameters:\n{scoring_data}")

    return scoring_data, updated_threshold_all


def update_scoring_params_from_dash(
    preagg_segment,
    preagg_joint,
    agg_method_segment,
    agg_method_joint,
    agg_weight_segment,
    agg_weight_joint,
    removal_threshold_type,
    removal_threshold_value,
    scoring_params_data,
):
    scoring_params = scoringParameters.ScoringParameters.model_validate_json(
        scoring_params_data
    )

    scoring_params.preagg_thresholding.segment = preagg_segment
    scoring_params.preagg_thresholding.joint = preagg_joint

    scoring_params.aggregation_method.segment = agg_method_segment
    scoring_params.aggregation_method.joint = agg_method_joint

    scoring_params.aggregation_weight.segment = agg_weight_segment
    scoring_params.aggregation_weight.joint = agg_weight_joint

    if removal_threshold_type == "all":
        scoring_params.removal_threshold = {
            marker: removal_threshold_value for marker in active_body.get_markers()
        }
    else:
        scoring_params.removal_threshold[removal_threshold_type] = (
            removal_threshold_value
        )

    updated_scoring_params_data = scoring_params.model_dump_json()
    return updated_scoring_params_data


def update_scoring_params_from_file(contents):
    content_type, content_string = contents.split(",")
    decoded = base64.b64decode(content_string)

    # Try to load the content into a dataframe, if it's a CSV or Excel
    # try:
    #     json_data = json.loads(decoded.decode("utf-8"))
    # except Exception as e:
    #     raise ValueError(f"Error loading JSON file: {e}")

    try:
        scoring_params = scoringParameters.ScoringParameters.model_validate_json(
            decoded
        )
    except pydantic.ValidationError as e:
        LOGGER.error(f"Error importing scoring parameters: {e}")
        return dash.no_update

    scoring_params, removal_threshold_all = refactor_removal_threshold(
        scoring_params, active_body.get_markers()
    )
    if ignore_symmetry:
        scoring_params = remove_symmetry_from_scoring(scoring_params)

    return scoring_params.model_dump_json(), removal_threshold_all


@app.callback(
    Output("download-scoring", "data"),
    Input("export-scoring", "n_clicks"),
    State("scoring-params", "data"),
    prevent_initial_call=True,
)
def export_scoring_params(n_clicks, scoring_params_data):
    if n_clicks:
        scoring_params = scoringParameters.ScoringParameters.model_validate_json(
            scoring_params_data
        )
        json_string = scoring_params.model_dump_json()

        return dcc.send_string(json_string, "scoring_parameters.json")

    raise dash.exceptions.PreventUpdate


def run():
    app.run_server(debug=True)


if __name__ == "__main__":
    run()
