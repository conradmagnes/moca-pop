"""

    Interactive Web-based Application to Test Scoring Strategies for Rigid Body Calibration
    =======================================================================================

    This script should user `dash` to create an interactive web-based application that allows
    users to move nodes in 3D space and see the connected segments and joints for each node.
    
    Layout:
    Middle: 3D plot showing nodes and segments
    Left side (20% width): List of connected segments and joints for the selected node and total score.
        - each component should display information about length/angle, threshold, and residual.
    Bottom: Sliders to move the selected node in the x, y, and z directions. Buttons to reset the selected node or all nodes.
    Right side (20%): Display scoring parameters and options for the user to adjust.
        - include buttons to import / export scoringParameters json file.

    Dynamic Updates:
    - When a node is selected, the connected segments and joints should be displayed. Sliders should be updated to show the current offset.
    - When a node is moved, the connected segments and joints should be updated.
    - When a node is reset, the connected segments and joints should be reset.
    - When a segment exceeds the threshold, the color should change to red.
    - When a node exceeds the score threshold, the color should change to red.

"""

import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import copy
import os

import numpy as np
import plotly.graph_objs as go

import mocap_popy.config.directory as directory
import mocap_popy.config.regex as regex
from mocap_popy.models.rigid_body import Node, Segment, Joint
from mocap_popy.utils import rigid_body_loader
from mocap_popy.scripts.unassign_rb_markers.scoring import scorer, scoringParameters


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
if ignore_symmetry and isinstance(scoring_params.removal_threshold, dict):
    scoring_params.removal_threshold = {
        regex.parse_symmetrical_component(k)[1]: v
        for k, v in scoring_params.removal_threshold.items()
    }


# Initial Node Positions
initial_positions = {m.marker: m.position for m in calibrated_body.nodes}

slider_params = {
    "min": -100,
    "max": 100,
    "step": 10,
    "value": 0,
    "marks": None,
    "tooltip": {"placement": "top"},
}
slider_style = {"width": "100%", "height": "100%"}


def get_slider(
    name: str, slider_params, slider_style: dict = None, div_style: dict = None
):
    ss = slider_style or {}
    ds = div_style or {}
    slider_id = f"slider-{'-'.join(name.lower().split(' '))}"
    return html.Div(
        [
            html.Label(f"{name}:"),
            html.Div(
                [dcc.Slider(id=slider_id, **slider_params)],
                style=slider_style,
            ),
        ],
        style=div_style,
    )


def get_scene_limits(positions, slider_params):
    scene_limits = {}
    for i, ax in enumerate(["x", "y", "z"]):
        min_val = min([pos[i] for pos in positions.values()]) + slider_params["min"]
        max_val = max([pos[i] for pos in positions.values()]) + slider_params["max"]
        scene_limits[f"{ax}axis"] = {"range": [min_val, max_val]}

    return scene_limits


scene_limits = get_scene_limits(initial_positions, slider_params)

browser_margin = 8


def subtract_browser_margin(value: str, browser_margin: int = 8, num_margins: int = 1):
    """! Get the relative view dimension based on the anchor and browser margin"""
    return f"calc({value} - {browser_margin * num_margins}px)"


def add_browser_margin(value: str, browser_margin: int = 8, num_margins: int = 1):
    """! Get the relative view dimension based on the anchor and browser margin"""
    return f"calc({value} + {browser_margin * num_margins}px)"


# Initialize the Dash app
app = dash.Dash(__name__, suppress_callback_exceptions=True)

outline_border = {"border": "1px solid black"}
standard_padding = {
    "box-sizing": "border-box",
    "padding-top": "1vh",
    "padding-right": "1vh",
    "padding-bottom": "1vh",
    "padding-left": "1vh",
}

graph_style = {
    "position": "absolute",
    "height": "70vh",
    "width": "50vw",
    "left": "25vw",
    "box-sizing": "border-box",
    **outline_border,
}
left_panel_style = {
    "position": "absolute",
    "height": subtract_browser_margin("100vh", num_margins=2),
    "width": subtract_browser_margin("25vw"),
    **standard_padding,
    **outline_border,
}
slider_panel_style = {
    "position": "absolute",
    "top": add_browser_margin("70vh"),
    "left": "25vw",
    "width": "50vw",
    "height": subtract_browser_margin("30vh", num_margins=2),
    **standard_padding,
    **outline_border,
}
slider_panel_label = "Set Offsets"

reset_button_style = {
    "position": "absolute",
    "height": "20%",
    "width": "30%",
}

NODE_OFFSET_DIV = html.Div(
    [
        html.Label(id="slider-label", children=slider_panel_label),
        html.Div(
            [
                get_slider("X", slider_params, slider_style),
                get_slider("Y", slider_params, slider_style),
                get_slider("Z", slider_params, slider_style),
            ],
            style={
                "height": "50%",
                "width": "50%",
                "left": "50%",
                "position": "absolute",
            },
        ),
        html.Button(
            "Reset",
            id="reset-button",
            n_clicks=0,
            style={
                "top": "25%",
                "left": "10%",
                **reset_button_style,
            },
        ),
        html.Button(
            "Reset All",
            id="reset-all-button",
            n_clicks=0,
            style={"top": "60%", "left": "10%", **reset_button_style},
        ),
    ],
    style=slider_panel_style,
)


# Layout of the App
app.layout = html.Div(
    [
        html.Div(id="connected-info", style=left_panel_style),
        dcc.Graph(
            id="3d-plot",
            style=graph_style,
        ),
        NODE_OFFSET_DIV,
        dcc.Store(id="node-positions", data=initial_positions),
        dcc.Store(id="selected-node", data=active_body.get_markers()[0]),
        dcc.Store(
            id="node-offsets", data={node: [0, 0, 0] for node in initial_positions}
        ),
        dcc.Store(id="max-score", data=0.0),
    ]
)


# Callback to update the 3D plot
@app.callback(
    Output("3d-plot", "figure"),
    Input("node-positions", "data"),
)
def update_graph(node_positions):
    # Update node positions
    for marker, position in node_positions.items():
        active_body.get_node(marker).position = np.array(position)

    # Create traces for nodes
    node_trace = go.Scatter3d(
        x=[n.position[0] for n in active_body.nodes],
        y=[n.position[1] for n in active_body.nodes],
        z=[n.position[2] for n in active_body.nodes],
        mode="markers+text",
        marker=dict(size=8, color="red"),
        text=active_body.get_markers(),
        textposition="top center",
    )

    # Create traces for segments
    segment_traces = []
    for seg in active_body.segments:
        segment_trace = go.Scatter3d(
            x=[seg.nodes[0].position[0], seg.nodes[1].position[0]],
            y=[seg.nodes[0].position[1], seg.nodes[1].position[1]],
            z=[seg.nodes[0].position[2], seg.nodes[1].position[2]],
            mode="lines",
            line=dict(color="black", width=2),
            hoverinfo="skip",
            hovertemplate=None,
        )
        segment_traces.append(segment_trace)

    # Combine traces
    data = [node_trace] + segment_traces

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
    Input("3d-plot", "clickData"),
)
def select_node(click_data):
    if click_data is None:
        return active_body.get_markers()[0]  # Default to "A" if nothing is clicked
    # Get the node name that was clicked
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
            return 0, 0, 0, f"{slider_panel_label}: {selected_node}"
        elif trigger == "reset-all-button":
            # Reset all nodes
            for node in node_offsets:
                node_offsets[node] = [0, 0, 0]
            return 0, 0, 0, f"{slider_panel_label}: {selected_node}"

    # Otherwise, use the stored offset for the selected node
    offset_x, offset_y, offset_z = node_offsets[selected_node]
    return offset_x, offset_y, offset_z, f"{slider_panel_label}: {selected_node}"


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

    # Determine if "Reset All" was clicked
    if ctx.triggered and ctx.triggered[0]["prop_id"] == "reset-all-button.n_clicks":
        # Reset all node positions and offsets
        updated_positions = {
            node: initial_positions[node].tolist() for node in initial_positions
        }
        updated_offsets = {node: [0, 0, 0] for node in initial_positions}
    else:
        # Update the offset of the selected node
        updated_offsets[selected_node] = [slider_x, slider_y, slider_z]

        # Update the position of the selected node using the offset
        initial_position = initial_positions[selected_node]  # Get the original position
        updated_positions[selected_node] = [
            initial_position[0] + slider_x,
            initial_position[1] + slider_y,
            initial_position[2] + slider_z,
        ]
    active_body.get_node(selected_node).position = np.array(
        updated_positions[selected_node]
    )

    return updated_positions, updated_offsets


# Callback to display connected segments and joints
@app.callback(
    [Output("connected-info", "children"), Output("max-score", "data")],
    [
        Input("selected-node", "data"),
        Input("slider-x", "value"),
        Input("slider-y", "value"),
        Input("slider-z", "value"),
        Input("reset-all-button", "n_clicks"),
    ],
    State("max-score", "data"),
)
def update_connected_info(
    selected_node, slider_x, slider_y, slider_z, reset_all_clicks, max_score
):
    # Get connected segments and joints
    connected_segments = active_body.get_connected_segments(selected_node)
    for seg in connected_segments:
        seg.compute_length()
        seg.compute_residuals(calibrated_body)

    connected_joints = active_body.get_connected_joints(selected_node)
    for joint in connected_joints:
        joint.compute_angle()
        joint.compute_residuals(calibrated_body)

    agg_segment = active_body.get_aggregate_segment_residuals(
        selected_node,
        ["residual_calib"],
        scoring_params.aggregation_method.segment,
        (not scoring_params.preagg_thresholding.segment),
    )
    agg_joint = active_body.get_aggregate_joint_residuals(
        selected_node,
        ["residual_calib"],
        scoring_params.aggregation_method.joint,
        (not scoring_params.preagg_thresholding.joint),
    )

    raw_score = (
        scoring_params.aggregation_weight.segment * agg_segment
        + scoring_params.aggregation_weight.joint * agg_joint
    )
    threshold = scoring_params.removal_threshold
    if (
        isinstance(threshold, dict)
        and selected_node in scoring_params.removal_threshold
    ):
        threshold = scoring_params.removal_threshold[selected_node]

    score = max(0, raw_score - threshold)
    if score > max_score:
        max_score = score

    connected_segment_header = html.Tr(
        [
            html.Th("Segment", style={"text-align": "left"}),
            html.Th("L (mm)", style={"text-align": "right"}),
            html.Th("Tol", style={"text-align": "right"}),
            html.Th("Res", style={"text-align": "right"}),
        ]
    )
    connected_segments_rows = [
        html.Tr(
            [
                html.Td(str(seg)),
                html.Td(f"{seg.length:.1f}", style={"text-align": "right"}),
                html.Td(f"{seg.tolerance:.2f}", style={"text-align": "right"}),
                html.Td(f"{seg.residual_calib:.1f}", style={"text-align": "right"}),
            ]
        )
        for seg in connected_segments
    ]
    connected_segments_rows.append(
        html.Tr(
            [
                html.Th("Aggregated", style={"text-align": "right"}),
                html.Td(""),
                html.Td(""),
                html.Td(
                    f"{agg_segment:.1f}",
                    style={"text-align": "right", "font-weight": "bold"},
                ),
            ]
        )
    )

    # Construct table rows for connected joints
    connected_joint_header = html.Tr(
        [
            html.Th("Name", style={"text-align": "left"}),
            html.Th("Ang (deg)", style={"text-align": "right"}),
            html.Th("Tol", style={"text-align": "right"}),
            html.Th("Res", style={"text-align": "right"}),
        ]
    )
    connected_joints_rows = [
        html.Tr(
            [
                html.Td(str(joint)),
                html.Td(f"{joint.angle:.0f}", style={"text-align": "right"}),
                html.Td(f"{joint.tolerance:.0f}", style={"text-align": "right"}),
                html.Td(f"{joint.residual_calib:.1f}", style={"text-align": "right"}),
            ]
        )
        for joint in connected_joints
    ]
    connected_joints_rows.append(
        html.Tr(
            [
                html.Th("Aggregated", style={"text-align": "right"}),
                html.Td(""),
                html.Td(""),
                html.Td(
                    f"{agg_joint:.1f}",
                    style={"text-align": "right", "font-weight": "bold"},
                ),
            ]
        )
    )

    # Create scrollable tables for segments and joints
    connected_info = html.Div(
        [
            html.H2(f"Selected Node: {selected_node}"),
            html.H4("Connected Segments:"),
            html.Div(
                html.Table(
                    [
                        html.Thead(connected_segment_header),
                        html.Tbody(connected_segments_rows),
                    ],
                    style={"width": "100%"},
                ),
                style={
                    "height": "150px",  # Set a fixed height for scrolling
                    "overflow-y": "auto",  # Enable vertical scrolling when content exceeds height
                    "border": "1px solid black",  # Border for visibility
                    "margin-bottom": "20px",  # Space below the table
                },
            ),
            html.H4("Connected Joints:"),
            html.Div(
                html.Table(
                    [
                        html.Thead(connected_joint_header),
                        html.Tbody(connected_joints_rows),
                    ],
                    style={"width": "100%"},
                ),
                style={
                    "height": "200px",  # Set a fixed height for scrolling
                    "overflow-y": "auto",  # Enable vertical scrolling when content exceeds height
                    "border": "1px solid black",  # Border for visibility
                    "margin-bottom": "20px",  # Space below the table
                },
            ),
            html.H4(f"Raw score: {raw_score:.1f}  |   Threshold: {threshold:.1f}"),
            html.H3(f"Score: {score:.1f}"),
            html.H3(f"Max Score: {max_score:.1f}"),
        ],
    )

    return connected_info, max_score


if __name__ == "__main__":
    app.run_server(debug=True)
