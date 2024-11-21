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
from mocap_popy.models.rigid_body import Node, Segment, Joint
from mocap_popy.utils import rigid_body_loader


project_dir = os.path.join(directory.DATASET_DIR, "shoe_stepping")
vsk_fp = os.path.join(project_dir, "subject.vsk")
calibrated_rigid_bodies = rigid_body_loader.get_rigid_bodies_from_vsk(vsk_fp)

rb_name = "Right_Foot"
nodes_to_vis = ["Right_TM", "Right_TL", "Right_TF"]

calibrated_body = calibrated_rigid_bodies[rb_name]
calibrated_body.compute_segment_lengths()
calibrated_body.compute_joint_angles()
active_body = copy.deepcopy(calibrated_body)

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


def get_scene_limits(positions, slider_params):
    scene_limits = {}
    for i, ax in enumerate(["x", "y", "z"]):
        min_val = min([pos[i] for pos in positions.values()]) + slider_params["min"]
        max_val = max([pos[i] for pos in positions.values()]) + slider_params["max"]
        scene_limits[f"{ax}axis"] = {"range": [min_val, max_val]}

    return scene_limits


scene_limits = get_scene_limits(initial_positions, slider_params)

# Initialize the Dash app
app = dash.Dash(__name__, suppress_callback_exceptions=True)

# Layout of the App
app.layout = html.Div(
    [
        dcc.Graph(id="3d-plot", style={"height": "70vh"}),
        dcc.Store(id="node-positions", data=initial_positions),
        dcc.Store(id="selected-node", data=active_body.get_markers()[0]),
        dcc.Store(
            id="node-offsets", data={node: [0, 0, 0] for node in initial_positions}
        ),
        html.Div(
            [
                html.Label(id="slider-label", children="Move Node A:"),
                html.Div(
                    [
                        html.Label("X:"),
                        dcc.Slider(id="slider-x", **slider_params),
                    ]
                ),
                html.Div(
                    [
                        html.Label("Y:"),
                        dcc.Slider(id="slider-y", **slider_params),
                    ]
                ),
                html.Div(
                    [
                        html.Label("Z:"),
                        dcc.Slider(id="slider-z", **slider_params),
                    ]
                ),
                html.Button(
                    "Reset", id="reset-button", n_clicks=0
                ),  # Reset button for current node
                html.Button(
                    "Reset All", id="reset-all-button", n_clicks=0
                ),  # Reset all nodes button
            ]
        ),
        html.Div(id="connected-info"),  # Display connected segments and joints
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
        )
        segment_traces.append(segment_trace)

    # Combine traces
    data = [node_trace] + segment_traces

    # Layout for 3D plot
    layout = go.Layout(
        margin=dict(l=0, r=0, b=0, t=0),
        scene={"aspectmode": "cube", **scene_limits},
        uirevision="same",
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
            return 0, 0, 0, f"Move Node {selected_node}:"
        elif trigger == "reset-all-button":
            # Reset all nodes
            for node in node_offsets:
                node_offsets[node] = [0, 0, 0]
            return 0, 0, 0, f"Move Node {selected_node}:"

    # Otherwise, use the stored offset for the selected node
    offset_x, offset_y, offset_z = node_offsets[selected_node]
    return offset_x, offset_y, offset_z, f"Move Node {selected_node}:"


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
    Output("connected-info", "children"),
    [
        Input("selected-node", "data"),
        Input("slider-x", "value"),
        Input("slider-y", "value"),
        Input("slider-z", "value"),
        Input("reset-all-button", "n_clicks"),
    ],
)
def update_connected_info(
    selected_node, slider_x, slider_y, slider_z, reset_all_clicks
):
    # Get connected segments and joints
    connected_segments = active_body.get_connected_segments(selected_node)
    for seg in connected_segments:
        seg.compute_length()
        seg.compute_residuals(calibrated_body)

    connected_segments_info = [
        f"Segment {seg}: length={seg.length:.2f}, residual={seg.residual_calib:.2f}"
        for seg in connected_segments
    ]

    # Combine all the connected information
    connected_info = html.Div(
        [
            html.H4(f"Connected Segments and Joints for Node {selected_node}:"),
            html.Ul([html.Li(info) for info in connected_segments_info]),
        ]
    )

    return connected_info


if __name__ == "__main__":
    app.run_server(debug=True)
