"""

    Interactive Web-based Application to Test Scoring Strategies for Rigid Body Calibration
    =======================================================================================

    This script should user `dash` to create an interactive web-based application that allows
    users to move nodes in 3D space and see the connected segments and joints for each node.
    
    @author C. McCarthy
"""

import argparse
import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State, ALL, MATCH
import copy
import os
import logging
import base64
import webbrowser
from threading import Timer

import numpy as np
import plotly.graph_objs as go
import pydantic

import mocap_popy.config.directory as directory
import mocap_popy.config.regex as regex
import mocap_popy.config.logger as logger
from mocap_popy.models.rigid_body import RigidBody
from mocap_popy.utils import (
    rigid_body_loader,
    c3d_parser,
    hmi,
    model_template_loader,
    json_utils,
)
from mocap_popy.scripts.unassign_rb_markers.scoring import scorer, scoringParameters

import mocap_popy.aux_scripts.interactive_score_analyzer.constants as isa_consts
import mocap_popy.aux_scripts.interactive_score_analyzer.layout as isa_layout
import mocap_popy.aux_scripts.interactive_score_analyzer.helpers as isa_helpers

LOGGER = logging.getLogger("InteractiveScoreAnalyzer")


def open_browser():
    """!Open the default web browser to the default Dash server."""
    webbrowser.open_new("http://127.0.0.1:8050/")


def run(
    calibrated_body: RigidBody,
    active_body: RigidBody,
    scoring_params: scoringParameters.ScoringParameters = None,
    ignore_symmetry: bool = False,
):
    if not scoring_params:
        scoring_params = scoringParameters.ScoringParameters()

    scoring_params, removal_threshold_all = isa_helpers.refactor_removal_threshold(
        scoring_params, calibrated_body.get_markers()
    )
    if ignore_symmetry:
        scoring_params = isa_helpers.remove_symmetry_from_scoring(scoring_params)

    missing_nodes = [
        m for m in calibrated_body.get_markers() if not active_body.get_node(m).exists
    ]
    button_states = {str(idx): "Fit" for idx in range(len(missing_nodes))}

    component_scores = {
        comp_name: {m.marker: 0 for m in calibrated_body.nodes}
        for comp_name in ["segment", "joint"]
    }

    # Create Dash app
    app = dash.Dash(__name__, suppress_callback_exceptions=True)

    # Create variables to store data
    stores = [
        dcc.Store(id="selected-node", data=active_body.get_markers()[0]),
        dcc.Store(id="node-moved", data=False),
        dcc.Store(
            id="node-positions", data={m.marker: m.position for m in active_body.nodes}
        ),
        dcc.Store(
            id="node-offsets",
            data={node: [0, 0, 0] for node in active_body.get_markers()},
        ),
        dcc.Store(id="component-scores", data=component_scores),
        dcc.Store(id="raw-node-scores", data={m: 0 for m in active_body.get_markers()}),
        dcc.Store(id="node-scores", data={m: 0 for m in active_body.get_markers()}),
        dcc.Store(id="max-score", data=0.0),
        dcc.Store(id="max-marker", data="N/A"),
        dcc.Store(id="scoring-params", data=scoring_params.model_dump_json()),
        dcc.Store(id="removal-threshold-all", data=removal_threshold_all),
        dcc.Store(id="removed-nodes", data={}),
        dcc.Store(id="staged-for-addback", data={}),
        dcc.Store(
            id="initial-positions",
            data={m.marker: m.position for m in active_body.nodes},
        ),
        dcc.Store(id="missing-nodes", data=missing_nodes),
        dcc.Store(id="fit-button-states", data=button_states),
    ]

    # Create layout
    app.layout = html.Div(
        [
            isa_layout.generate_ni_div(),
            isa_layout.generate_ig(),
            isa_layout.generate_nm_div(None, missing_nodes),
            isa_layout.generate_spe_div(scoring_params, removal_threshold_all),
            *stores,
        ]
    )

    # Callback to update the 3D plot
    @app.callback(
        Output("interactive-graph", "figure"),
        [
            Input("node-positions", "data"),
            Input("node-scores", "data"),
            Input("label-toggle", "value"),
        ],
    )
    def update_graph(node_positions, node_scores, label_toggle):
        hide_labels = "hide" in label_toggle
        active_mode = "markers" if hide_labels else "markers+text"

        calib_node_trace = isa_helpers.generate_node_trace(
            calibrated_body,
            node_scores,
            "markers",
            dict(size=5),
        )
        active_node_trace = isa_helpers.generate_node_trace(
            active_body,
            node_scores,
            active_mode,
            dict(size=7),
            body_type="active",
        )

        calib_segment_traces = isa_helpers.generate_segment_traces(
            calibrated_body, dict(width=1)
        )
        active_segment_traces = isa_helpers.generate_segment_traces(
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

        scene_limits = isa_layout.get_scene_limits(
            existing_pos, isa_layout.NM_SLIDER_ARGS
        )

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
        [
            Output({"type": "fit-button", "index": ALL}, "children"),
            Output("fit-button-states", "data"),
        ],
        [
            Input({"type": "fit-button", "index": ALL}, "n_clicks"),
            Input("reset-all-button", "n_clicks"),
        ],
        [
            State("fit-button-states", "data"),
            State("missing-nodes", "data"),
        ],
    )
    def toggle_fit_button_text(
        n_clicks_list, reset_all_clicks, button_states, missing_nodes: list
    ):
        if button_states is None:
            button_states = {}

        ctx = dash.callback_context
        trigger = ctx.triggered_id if ctx.triggered_id else None

        if trigger == "reset-all-button":
            button_states = {index: "Fit" for index in button_states}
            return ["Fit"] * len(n_clicks_list), button_states

        elif isinstance(trigger, dict) and trigger.get("type") == "fit-button":
            index = str(missing_nodes.index(trigger["index"]))

            if index not in button_states or button_states[index] == "Fit":
                button_states[index] = "Delete"
            else:
                button_states[index] = "Fit"

            updated_texts = [
                button_states.get(str(idx), "Fit") for idx in range(len(n_clicks_list))
            ]
            return updated_texts, button_states

        raise dash.exceptions.PreventUpdate

    @app.callback(
        [
            Output("removed-nodes", "data"),
            Output("staged-for-addback", "data"),
            Output("initial-positions", "data"),
        ],
        [
            Input("remove-node-button", "n_clicks"),
            Input({"type": "add-back-button", "index": ALL}, "n_clicks"),
        ],
        [
            State("selected-node", "data"),
            State("removed-nodes", "data"),
            State("staged-for-addback", "data"),
            State("initial-positions", "data"),
        ],
    )
    def handle_node_removal(
        remove_node_clicks,
        n_clicks_list,
        selected_node,
        removed_nodes,
        staged_for_addback,
        initial_positions,
    ):
        ctx = dash.callback_context
        updated_removed_nodes = removed_nodes.copy()
        updated_initial_positions = initial_positions.copy()

        if ctx.triggered_id == "remove-node-button":
            if selected_node in removed_nodes:
                raise dash.exceptions.PreventUpdate
            updated_removed_nodes[selected_node] = updated_initial_positions[
                selected_node
            ].copy()
            updated_initial_positions[selected_node] = [0, 0, 0]
            if selected_node in staged_for_addback:
                staged_for_addback.pop(selected_node)
            return updated_removed_nodes, dash.no_update, updated_initial_positions

        for node_name, n_clicks in zip(removed_nodes.keys(), n_clicks_list):
            if n_clicks > 0:
                staged_for_addback[node_name] = removed_nodes.pop(node_name)
                updated_initial_positions[node_name] = staged_for_addback[
                    node_name
                ].copy()
                return removed_nodes, staged_for_addback, updated_initial_positions

        raise dash.exceptions.PreventUpdate

    @app.callback(
        Output("nm-removal", "children"),
        [
            Input("removed-nodes", "data"),
        ],
    )
    def update_removed_node_table(removed_nodes):
        return isa_layout.generate_nm_removal_div_children(removed_nodes)

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
        slider_x,
        slider_y,
        slider_z,
        reset_all_clicks,
        selected_node,
        node_offsets,
    ):
        ctx = dash.callback_context
        trigger_id = ctx.triggered_id if ctx.triggered_id else None
        if trigger_id == "reset-all-button":
            return {node: [0, 0, 0] for node in node_offsets}

        node_offsets[selected_node] = [slider_x, slider_y, slider_z]
        return node_offsets

    # Callback to update the sliders when a node is selected or when reset buttons are clicked
    @app.callback(
        [
            Output("slider-x", "value"),
            Output("slider-y", "value"),
            Output("slider-z", "value"),
            Output("slider-x", "disabled"),
            Output("slider-y", "disabled"),
            Output("slider-z", "disabled"),
            Output("slider-label", "children"),
            Output("remove-node-button", "disabled"),
        ],
        [
            Input("selected-node", "data"),
            Input("reset-button", "n_clicks"),
            Input("reset-all-button", "n_clicks"),
            Input("remove-node-button", "n_clicks"),
            Input({"type": "add-back-button", "index": ALL}, "n_clicks"),
        ],
        State("node-offsets", "data"),
        State("removed-nodes", "data"),
        State("missing-nodes", "data"),
    )
    def update_sliders(
        selected_node,
        reset_clicks,
        reset_all_clicks,
        remove_node_clicks,
        add_back_clicks,
        node_offsets,
        removed_nodes,
        missing_nodes,
    ):
        ctx = dash.callback_context
        trigger_id = ctx.triggered_id if ctx.triggered_id else None
        if isinstance(trigger_id, dict) and trigger_id.get("type") == "add-back-button":
            for n_clicks, add_back_id in zip(add_back_clicks, ctx.inputs_list[4]):
                if n_clicks > 0 and selected_node == add_back_id["id"]["index"]:
                    return (
                        0,
                        0,
                        0,
                        False,
                        False,
                        False,
                        dash.no_update,
                        dash.no_update,
                    )
            raise dash.exceptions.PreventUpdate
        if trigger_id in ["reset-all-button", "reset-button"]:
            return (
                0,
                0,
                0,
                dash.no_update,
                dash.no_update,
                dash.no_update,
                dash.no_update,
                dash.no_update,
            )
        if trigger_id == "remove-node-button":
            return (0, 0, 0, True, True, True, dash.no_update, dash.no_update)

        disable_sliders = [False, False, False]
        disable_button = False
        if selected_node in removed_nodes or selected_node in missing_nodes:
            disable_sliders = [True, True, True]
            disable_button = True

        return (
            *node_offsets[selected_node],
            *disable_sliders,
            f"{isa_layout.NM_HEADER_LABEL}: {selected_node}",
            disable_button,
        )

    # Combined Callback to update node positions, offsets, and handle "Reset All" functionality
    @app.callback(
        [Output("node-positions", "data"), Output("node-moved", "data")],
        [
            Input("node-offsets", "data"),
            Input("removed-nodes", "data"),
            Input("reset-all-button", "n_clicks"),
            Input({"type": "fit-button", "index": ALL}, "n_clicks"),
        ],
        [
            State("selected-node", "data"),
            State("node-positions", "data"),
            State("staged-for-addback", "data"),
            State("initial-positions", "data"),
            State("missing-nodes", "data"),
            State("fit-button-states", "data"),
        ],
    )
    def update_node_positions(
        node_offsets,
        removed_nodes,
        reset_all_clicks,
        fit_clicks_list,
        selected_node,
        node_positions,
        staged_for_addback,
        initial_positions,
        missing_nodes,
        button_states,
    ):
        ctx = dash.callback_context
        trigger = ctx.triggered[0] if ctx.triggered else None

        # Check which input triggered the callback
        if not trigger:
            raise dash.exceptions.PreventUpdate

        updated_positions = node_positions.copy()
        prop_id = trigger["prop_id"]

        if "node-offsets" in prop_id:
            if selected_node in missing_nodes:
                raise dash.exceptions.PreventUpdate

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

        elif "reset-all-button" in prop_id:
            updated_positions = {
                node: np.array(initial_positions[node])
                for node in initial_positions
                if active_body.get_node(node).exists
            }
            active_body.update_node_positions(
                updated_positions, recompute_lengths=True, recompute_angles=True
            )
            active_body.compute_segment_residuals(calibrated_body)
            active_body.compute_joint_residuals(calibrated_body)

        elif "removed-nodes" in prop_id:
            for node, pos in staged_for_addback.items():
                updated_positions[node] = pos

            for node, pos in removed_nodes.items():
                updated_positions[node] = initial_positions[node]

            updated_positions = {m: np.array(p) for m, p in updated_positions.items()}
            active_body.update_node_positions(
                updated_positions, recompute_lengths=True, recompute_angles=True
            )
            active_body.compute_segment_residuals(calibrated_body)
            active_body.compute_joint_residuals(calibrated_body)

        elif "fit-button" in prop_id:
            trigger_id = ctx.triggered_id if ctx.triggered_id else None
            if (
                not isinstance(trigger_id, dict)
                or trigger_id.get("type") != "fit-button"
            ):
                raise dash.exceptions.PreventUpdate

            index = missing_nodes.index(trigger_id["index"])
            n_clicks = fit_clicks_list[index]
            node_name = missing_nodes[index]
            key = str(index)

            if n_clicks > 0 and key in button_states:
                if button_states[key] == "Fit":
                    pos = calibrated_body.get_node(node_name).position.copy()
                else:
                    pos = np.array([0, 0, 0])
                updated_positions[node_name] = pos
                active_body.update_node_positions(
                    {node_name: pos}, recompute_lengths=True, recompute_angles=True
                )
                active_body.compute_segment_residuals(calibrated_body)
                active_body.compute_joint_residuals(calibrated_body)

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
        connected_segments = active_body.get_connected_segments(
            selected_node, only_existing=True
        )
        connected_joints = active_body.get_connected_joints(
            selected_node, only_existing=True
        )

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

        try:
            scoring_params = scoringParameters.ScoringParameters.model_validate_json(
                decoded
            )
        except pydantic.ValidationError as e:
            LOGGER.error(f"Error importing scoring parameters: {e}")
            return dash.no_update

        scoring_params, removal_threshold_all = isa_helpers.refactor_removal_threshold(
            scoring_params, active_body.get_markers()
        )
        if ignore_symmetry:
            scoring_params = isa_helpers.remove_symmetry_from_scoring(scoring_params)

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

    Timer(1, open_browser).start()
    app.run_server(debug=False)


def validate_offline_args(args) -> tuple:
    """!Validate offline mode arguments.
    Mainly checks file paths.

    @return tuple of project_dir, vsk_fp, subject_name, trial_fp
    """

    if not args.project_name:
        LOGGER.error("Project directory must be provided in offline mode.")
        exit(-1)

    if "example_datasets" in args.project_name:
        project_dir = os.path.join(
            directory.DATASET_DIR, args.project_name.split(os.sep)[-1]
        )
    elif args.project_name in os.listdir(directory.DATASET_DIR):
        LOGGER.info("Found project directory in example datasets.")
        project_dir = os.path.join(directory.DATASET_DIR, args.project_name)
    else:
        project_dir = args.project_name

    if not os.path.isdir(project_dir):
        LOGGER.error(f"Project directory does not exist ({project_dir}). Exiting.")
        exit(-1)

    vsk_fp = os.path.join(project_dir, f"{args.subject_name}.vsk")
    if not os.path.isfile(vsk_fp):
        LOGGER.error(f"VSK file was not found: {vsk_fp}. Exiting.")
        exit(-1)

    if args.trial_name == "":
        LOGGER.info("Trial name no provided, continuing with only VSK file.")
        trial_fp = None
    else:
        trial_fp = os.path.join(project_dir, f"{args.trial_name}.c3d")
        if not os.path.isfile(trial_fp):
            ui = hmi.get_user_input(
                "No trial file found. Continue without trial file? (y/n)",
                exit_on_quit=True,
                choices=[hmi.YES_KEYWORDS, hmi.NO_KEYWORDS],
                num_tries=5,
            )
            if ui == 1:
                LOGGER.error(f"Exiting.")
                exit(0)
            trial_fp = None

    return project_dir, vsk_fp, args.subject_name, trial_fp


def validate_online_args(args, vicon):
    """!Validate online mode arguments.

    Checks loaded trial and subject names, and vsk file.

    @param vicon ViconNexus instance.
    @return tuple of project_dir, vsk_fp, subject_name, trial_fp
    """
    project_dir, trial_name = vicon.GetTrialName()
    if not trial_name:
        ui = hmi.get_user_input(
            "No trial loaded. Coninue without trial? (y/n)",
            exit_on_quit=True,
            choices=[hmi.YES_KEYWORDS, hmi.NO_KEYWORDS],
            num_tries=5,
        )
        if ui == 1:
            LOGGER.error("Exiting.")
        trial_fp = None
    else:
        trial_fp = os.path.join(project_dir, f"{trial_name}.c3d")

    subject_names, subject_templates, subject_statuses = vicon.GetSubjectInfo()

    if not args.subject_name:
        LOGGER.info("Searching for available subject templates...")
        candidate_subject_names = []
        mapper = model_template_loader.load_mapper()
        for name, template, status in zip(
            subject_names, subject_templates, subject_statuses
        ):
            if template in mapper:
                candidate_subject_names.append((name, status))
        if len(candidate_subject_names) == 0:
            LOGGER.error("No matching templates found for trial subjects. Exiting.")
            exit(-1)
        elif len(candidate_subject_names) > 1:
            candidate_subject_names = [
                sn for sn, status in candidate_subject_names if status
            ]
            if len(candidate_subject_names) > 1 or len(candidate_subject_names) == 0:
                LOGGER.error(
                    "Could not infer subject name (multiple or none active). Please add subject name flag."
                )
                exit(-1)

        subject_name, _ = candidate_subject_names[0]

    elif args.subject_name not in subject_names:
        LOGGER.error(f"Subject name '{args.subject_name}' not found in Nexus. Exiting.")
        exit(-1)
    else:
        subject_name = args.subject_name

    vsk_fp = os.path.join(project_dir, f"{subject_name}.vsk")
    if not os.path.isfile(vsk_fp):
        LOGGER.error(f"VSK file was not found: {vsk_fp}. Exiting.")
        exit(-1)

    return project_dir, vsk_fp, subject_name, trial_fp


def validate_scoring_path(scoring_fp: str):
    """!Try loading the scoring path. Returns default if not found.

    @param scoring_path Path to scoring configuration file.
    """
    try:
        params = json_utils.import_json_as_str(scoring_fp)
        return scoringParameters.ScoringParameters.model_validate_json(params)
    except (
        FileNotFoundError,
        json_utils.json.JSONDecodeError,
        pydantic.ValidationError,
    ) as e:
        LOGGER.error(f"Error loading scoring parameters at {scoring_fp}. {e}")

    LOGGER.info("Continuing with default scoring parameters.")
    return scoringParameters.ScoringParameters()


def load_scoring_parameters(scoring_name: str = None) -> str:
    """!Load scoring parameters from a file.

    @param scoring_name Name or path of the scoring configuration file
    @return ScoringParameters object.
    """
    if not scoring_name:
        return scoringParameters.ScoringParameters()
    if os.path.isfile(scoring_name):
        return validate_scoring_path(scoring_name)

    scoring_dir = os.path.join(
        directory.SCRIPTS_DIR, "unassign_rb_markers", "scoring", "saved_parameters"
    )
    fn = scoring_name if scoring_name.endswith(".json") else f"{scoring_name}.json"
    if os.path.isfile(os.path.join(scoring_dir, fn)):
        return validate_scoring_path(os.path.join(scoring_dir, fn))

    LOGGER.warning(
        f"Scoring configuration file not found: {scoring_name}. Continuing with defaults."
    )
    return scoringParameters.ScoringParameters()


def load_calibrated_body(vsk_fp: str, ignore_symmetry: bool, rb_name: str) -> RigidBody:
    """!Load the rigid body from the VSK file.

    If no rigid body name is provided, the user is prompted to select one.

    @param vsk_fp Path to the VSK file.
    @param ignore_symmetry Ignore symmetry labels in the Rigid Body.

    @return RigidBody object.
    """
    calibrated_rigid_bodies = rigid_body_loader.get_rigid_bodies_from_vsk(
        vsk_fp, ignore_marker_symmetry=ignore_symmetry
    )
    available_rigid_bodies = list(calibrated_rigid_bodies.keys())
    if len(available_rigid_bodies) == 0:
        LOGGER.error("No rigid bodies found in the VSK file. Exiting.")
        exit(-1)
    if not rb_name or rb_name not in available_rigid_bodies:
        choices = ["[0] Exit"] + [
            f"[{i+1}] {rb}" for i, rb in enumerate(available_rigid_bodies)
        ]
        LOGGER.info("Available rigid bodies:\n" + "\n".join(choices))
        ui = hmi.get_user_input(
            "Select a rigid body to analyze:",
            choices=[(str(i),) for i in range(len(choices))],
            exit_on_quit=True,
        )
        if ui == 0:
            LOGGER.error("Exiting.")
            exit(0)
        rb_name = available_rigid_bodies[ui - 1]

    calibrated_body = calibrated_rigid_bodies[rb_name]
    calibrated_body.compute_segment_lengths()
    calibrated_body.compute_joint_angles()
    return calibrated_body


def load_active_body(
    calibrated_body: RigidBody,
    ignore_symmetry: bool,
    rb_name: str,
    trial_fp: str = None,
    frame: int = 0,
):
    """!Load the active body from the trial file.
    If no trial file is provided, the active body is the same as the calibrated body.
    If trial is found, calibrated body is transformed to match pose of the active body.

    @param calibrated_body RigidBody object.
    @param ignore_symmetry Ignore symmetry labels in the Rigid Body.
    @param rb_name Name of the rigid body to analyze.
    @param trial_fp Path to the trial file.
    @param frame Frame to analyze.
    """
    if not trial_fp:
        return copy.deepcopy(calibrated_body)

    c3d_reader = c3d_parser.get_reader(trial_fp)
    marker_trajectories = c3d_parser.get_marker_trajectories(c3d_reader)
    frames = c3d_parser.get_frames(c3d_reader)
    if frame in frames:
        frame = frames.index(frame)
    elif frame < 0 or frame >= len(frames):
        LOGGER.info(f"Using last frame in trial: {frames[-1]}")
        frame = len(frames) - 1

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

    active_nodes = [
        traj.generate_node(m, frame) for m, traj in marker_trajectories.items()
    ]
    active_body = copy.deepcopy(calibrated_body)
    active_body.update_node_positions(
        {n.marker: n.position for n in active_nodes},
        recompute_lengths=True,
        recompute_angles=True,
    )
    active_body.compute_segment_residuals(calibrated_body)
    active_body.compute_joint_residuals(calibrated_body)

    isa_helpers.best_fit_transform(calibrated_body, active_body)

    return active_body


def configure_parser():
    parser = argparse.ArgumentParser(
        description="Interactive Web-based Application to Test Rigid Body Scoring Strategies"
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Increase logging verbosity to debug.",
    )

    parser.add_argument(
        "-off",
        "--offline",
        action="store_true",
        help="Processes c3d files offline rather than connected to Nexus.",
    )

    parser.add_argument(
        "-pn",
        "--project_name",
        type=str,
        default="",
        help="Path to the project directory. Must be provided if using 'offline' mode.",
    )
    parser.add_argument(
        "-sn",
        "--subject_name",
        type=str,
        default="",
        help="Name of the subject. Must be provided if using 'offline' mode. Locates the VSK file.",
    )
    parser.add_argument(
        "--rb_name",
        type=str,
        default="",
        help="Name of the rigid body to analyze.",
    )

    parser.add_argument(
        "-tn",
        "--trial_name",
        type=str,
        default="",
        help="Name of the trial (optional). Uses marker trajectories from the trial to set the active body.",
    )

    parser.add_argument(
        "--scoring_name",
        type=str,
        default="",
        help="Name of the scoring configuration to use (saved to mocap_popy/scripts/unassign_rb_markers/scoring)",
    )

    parser.add_argument(
        "--ignore_symmetry",
        action="store_true",
        help="Ignore symmetry labels in the Rigid Body.",
    )

    parser.add_argument(
        "--frame",
        type=int,
        default=0,
        help="Frame to analyze (if trial provided).",
    )

    return parser


def main():
    parser = configure_parser()
    args = parser.parse_args()

    # mode = "w" if args.log else "off"
    logger.set_root_logger(name="interactive_score_analyzer", mode="off")
    werkzeug_logger = logging.getLogger("werkzeug")
    werkzeug_logger.setLevel(logging.ERROR)

    if args.verbose:
        LOGGER.setLevel(logging.DEBUG)
        isa_layout.LOGGER.setLevel(logging.DEBUG)

    if args.offline:
        vicon = None
        project_dir, vsk_fp, subject_name, trial_fp = validate_offline_args(args)
    else:
        try:
            from viconnexusapi import ViconNexus
        except ImportError as e:
            LOGGER.error(f"Error importing Vicon Nexus. Check installation. {e}")
            exit(-1)

        vicon = ViconNexus.ViconNexus()
        project_dir, vsk_fp, subject_name, trial_fp = validate_online_args(args, vicon)

    rb_name = args.rb_name
    ignore_symmetry = args.ignore_symmetry
    frame = args.frame

    desc_str = "Project: {}, VSK: {}".format(
        *[os.path.basename(x) for x in [project_dir, vsk_fp]]
    )
    if trial_fp:
        desc_str += ", Trial: {}, Frame: {}".format(os.path.basename(trial_fp), frame)
    LOGGER.info(desc_str)

    calibrated_body = load_calibrated_body(vsk_fp, ignore_symmetry, rb_name)
    active_body = load_active_body(
        calibrated_body, ignore_symmetry, calibrated_body.name, trial_fp, frame
    )

    scoring_params = load_scoring_parameters(args.scoring_name)

    run(calibrated_body, active_body, scoring_params, ignore_symmetry)


if __name__ == "__main__":
    main()
