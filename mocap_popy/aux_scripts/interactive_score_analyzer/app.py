"""

    Interactive Web-based Application to Test Scoring Strategies for Rigid Body Calibration
    =======================================================================================

    This script should user `dash` to create an interactive web-based application that allows
    users to move nodes in 3D space and see the connected segments and joints for each node.
    
    Callback mapping brainstorm:
    select_node -> sys moves slider, update connected info
    (user moves slider, reset, reset_all) -> update node offset, position, update connected info
    (user moves slider, reset, reset_all, update) -> update scores, connected info

"""

import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import copy
import os
import json

import numpy as np
import plotly.graph_objs as go

import mocap_popy.config.directory as directory
import mocap_popy.config.regex as regex
from mocap_popy.models.rigid_body import Node, Segment, Joint, RigidBody
from mocap_popy.utils import rigid_body_loader
from mocap_popy.scripts.unassign_rb_markers.scoring import scorer, scoringParameters

import mocap_popy.aux_scripts.interactive_score_analyzer.constants as isa_consts
import mocap_popy.aux_scripts.interactive_score_analyzer.layout as isa_layout


## ---- ##


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
active_body = copy.deepcopy(calibrated_body)

scoring_params = scoringParameters.ScoringParameters()
if ignore_symmetry:
    if isinstance(scoring_params.removal_threshold, dict):
        scoring_params.removal_threshold = {
            regex.parse_symmetrical_component(k)[1]: v
            for k, v in scoring_params.removal_threshold.items()
        }
        removal_threshold_all = 0
    else:
        removal_threshold_all = scoring_params.removal_threshold
        scoring_params.removal_threshold = {
            m: scoring_params.removal_threshold for m in calibrated_body.get_markers()
        }

# Initial Node Positions
initial_positions = {m.marker: m.position for m in calibrated_body.nodes}

component_scores = {
    comp_name: {m.marker: 0 for m in calibrated_body.nodes}
    for comp_name in ["segment", "joint"]
}
raw_node_scores = {m: 0 for m in active_body.get_markers()}
node_scores = {m: 0 for m in active_body.get_markers()}


active_marker_colors = {
    m.marker: isa_consts.get_component_color(0) for m in active_body.nodes
}
active_segment_colors = {
    str(s): isa_consts.get_component_color(s.residual_calib)
    for s in active_body.segments
}

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
        margs = marker_args.copy()
    else:
        colors = [isa_consts.get_component_color(s) for s in exisiting_node_scores]
        margs = {**marker_args, "color": colors}

    return go.Scatter3d(
        x=[n.position[0] for n in body.nodes],
        y=[n.position[1] for n in body.nodes],
        z=[n.position[2] for n in body.nodes],
        mode=mode,
        marker=margs,
        text=body.get_markers(),
        textposition=text_position,
    )


def generate_segment_traces(body: RigidBody, line_args: dict, body_type="calibrated"):
    traces = []
    for seg in body.segments:
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
    dcc.Store(id="node-positions", data=initial_positions),
    dcc.Store(id="node-offsets", data={node: [0, 0, 0] for node in initial_positions}),
    dcc.Store(id="component-scores", data=component_scores),
    dcc.Store(id="raw-node-scores", data=raw_node_scores),
    dcc.Store(id="node-scores", data=node_scores),
    dcc.Store(id="max-score", data=0.0),
    dcc.Store(id="max-marker", data="N/A"),
    dcc.Store(id="scoring-params", data=scoring_params.model_dump_json()),
    dcc.Store(id="removal-threshold-all", data=removal_threshold_all),
    dcc.Store(id="active-marker-colors", data=active_marker_colors),
    dcc.Store(id="active-segment-colors", data=active_segment_colors),
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
    # Update node positions
    active_body.update_node_positions(
        {m: np.array(p) for m, p in node_positions.items()}
    )

    # Create traces for nodes
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

    # Create traces for segments
    calib_segment_traces = generate_segment_traces(calibrated_body, dict(width=1))
    active_segment_traces = generate_segment_traces(
        active_body, dict(width=2), body_type="active"
    )
    # Combine traces
    data = (
        [calib_node_trace]
        + calib_segment_traces
        + [active_node_trace]
        + active_segment_traces
    )

    scene_limits = isa_layout.get_scene_limits(
        node_positions, isa_layout.NM_SLIDER_ARGS
    )

    # Layout for 3D plot
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
    # Determine which button (if any) was clicked
    ctx = dash.callback_context
    if ctx.triggered:
        trigger = ctx.triggered[0]["prop_id"].split(".")[0]
        if trigger == "reset-button":
            # Reset only the selected node
            node_offsets[selected_node] = [0, 0, 0]
            return 0, 0, 0, f"{isa_layout.NM_HEADER_LABEL}: {selected_node}"
        elif trigger == "reset-all-button":
            # Reset all nodes
            for node in node_offsets:
                node_offsets[node] = [0, 0, 0]
            return 0, 0, 0, f"{isa_layout.NM_HEADER_LABEL}: {selected_node}"

    # Otherwise, use the stored offset for the selected node
    offset_x, offset_y, offset_z = node_offsets[selected_node]
    return (
        offset_x,
        offset_y,
        offset_z,
        f"{isa_layout.NM_HEADER_LABEL}: {selected_node}",
    )


# Combined Callback to update node positions, offsets, and handle "Reset All" functionality
@app.callback(
    [
        Output("node-positions", "data"),
        Output("node-offsets", "data"),
    ],
    [
        Input("slider-x", "value"),
        Input("slider-y", "value"),
        Input("slider-z", "value"),
        Input("reset-all-button", "n_clicks"),
    ],
    [
        State("selected-node", "data"),
        State("node-positions", "data"),
        State("node-offsets", "data"),
    ],
)
def update_node_positions_and_offsets(
    slider_x,
    slider_y,
    slider_z,
    reset_all_clicks,
    selected_node,
    node_positions,
    node_offsets,
):
    ctx = dash.callback_context

    updated_positions = node_positions.copy()
    updated_offsets = node_offsets.copy()

    if ctx.triggered and ctx.triggered[0]["prop_id"] == "reset-all-button.n_clicks":
        # Reset all node positions and offsets
        updated_positions = {
            node: initial_positions[node] for node in initial_positions
        }
        updated_offsets = {node: [0, 0, 0] for node in initial_positions}
        active_body.update_node_positions(
            updated_positions, recompute_lengths=True, recompute_angles=True
        )
        active_body.compute_segment_residuals(calibrated_body)
        active_body.compute_joint_residuals(calibrated_body)

    else:
        updated_offsets[selected_node] = [slider_x, slider_y, slider_z]

        initial_position = initial_positions[selected_node]  # Get the original position
        updated_positions[selected_node] = [
            initial_position[0] + slider_x,
            initial_position[1] + slider_y,
            initial_position[2] + slider_z,
        ]
        active_node = active_body.get_node(selected_node)
        active_node.position = np.array(updated_positions[selected_node])
        for seg in active_node.get_connected_segments(active_body.segments):
            seg.compute_length()
            seg.compute_residuals(calibrated_body)
        for joint in active_node.get_connected_joints(active_body.joints):
            joint.compute_angle()
            joint.compute_residuals(calibrated_body)

    return (updated_positions, updated_offsets)


@app.callback(
    [
        Output("component-scores", "data"),
        Output("raw-node-scores", "data"),
        Output("node-scores", "data"),
        Output("max-score", "data"),
        Output("max-marker", "data"),
    ],
    [
        Input("slider-x", "value"),
        Input("slider-y", "value"),
        Input("slider-z", "value"),
        Input("reset-all-button", "n_clicks"),
        Input("update-button", "n_clicks"),
        Input("scoring-params", "data"),
    ],
    [
        State("node-positions", "data"),
        State("node-offsets", "data"),
        State("scoring-params", "data"),
        State("component-scores", "data"),
        State("raw-node-scores", "data"),
        State("node-scores", "data"),
        State("max-score", "data"),
        State("max-marker", "data"),
    ],
)
def update_scores(
    slider_x,
    slider_y,
    slider_z,
    reset_all_clicks,
    update_button_clicks,
    scoring_params_input,
    node_positions,
    node_offsets,
    scoring_params_state,
    component_scores,
    raw_node_scores,
    node_scores,
    max_score,
    max_marker,
):
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
def update_connected_info(
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
    Output("scoring-params", "data"),
    [Input("update-button", "n_clicks")],
    [
        State("preagg-segment", "value"),
        State("preagg-joint", "value"),
        State("agg-method-segment", "value"),
        State("agg-method-joint", "value"),
        State("agg-weight-segment", "value"),
        State("agg-weight-joint", "value"),
        State("removal-threshold-type", "value"),
        State("removal-threshold-value", "value"),
        State("scoring-params", "data"),
    ],
)
def update_scoring_parameters(
    n_clicks,
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
    # Prevent callback from running before the button is clicked
    if n_clicks == 0:
        raise dash.exceptions.PreventUpdate

    # Load current scoring parameters
    scoring_params = scoringParameters.ScoringParameters.model_validate_json(
        scoring_params_data
    )

    # Update PreAggThresholding
    scoring_params.preagg_thresholding.segment = preagg_segment
    scoring_params.preagg_thresholding.joint = preagg_joint

    # Update AggregationMethod
    scoring_params.aggregation_method.segment = agg_method_segment
    scoring_params.aggregation_method.joint = agg_method_joint

    # Update AggregationWeight
    scoring_params.aggregation_weight.segment = agg_weight_segment
    scoring_params.aggregation_weight.joint = agg_weight_joint

    # Update Removal Threshold
    if removal_threshold_type == "all":
        scoring_params.removal_threshold = {
            marker: removal_threshold_value for marker in active_body.get_markers()
        }
    else:
        scoring_params.removal_threshold[removal_threshold_type] = (
            removal_threshold_value
        )

    updated_scoring_params_data = scoring_params.model_dump_json()
    print(f"Updated Scoring Parameters:\n{updated_scoring_params_data}")
    return updated_scoring_params_data


def run():
    app.run_server(debug=True)


if __name__ == "__main__":
    run()
