"""!
    Layout Components and Style for Interactive Score Analyzer
    ==========================================================

    The app consists of 4 main panels with information:
    - (Left) Node Inspector, 'NI': 
        Shows info about the segments and joints connected to a selected node, and the node's score.
    - (Center Top) Interactive Graph, 'IG': 
        3D plot of nodes and segments. Nodes can be selected for inspection.
    - (Center Bottom) Node Manipulator, 'NM':
        Contains sliders and button to change the position of a node or remove it.
    - (Right) Scoring Parameter Editor, 'SPE':
        Options for importing, exporting, and changing scoring parameters.

    @author C. McCarthy
"""

import copy
import os
import json
from typing import Union
import logging

import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import numpy as np
import plotly.graph_objs as go

import mocap_popy.config.directory as directory
import mocap_popy.config.regex as regex
from mocap_popy.models.rigid_body import Node, Segment, Joint
from mocap_popy.utils import rigid_body_loader
from mocap_popy.scripts.unassign_rb_markers.scoring import scorer, scoringParameters
from mocap_popy.aux_scripts.interactive_score_analyzer import constants as isa_consts


LOGGER = logging.getLogger("ISALayout")

## ------ Margins ------ ##


def subtract_browser_margin(
    value: str, browser_margin: int = None, num_margins: int = 1
) -> str:
    """! Get the relative view dimension based on the anchor and browser margin"""
    browser_margin = browser_margin or isa_consts.BROSWER_MARGINS
    return f"calc({value} - {browser_margin * num_margins}px)"


def add_browser_margin(
    value: str, browser_margin: int = None, num_margins: int = 1
) -> str:
    """! Get the relative view dimension based on the anchor and browser margin"""
    browser_margin = browser_margin or isa_consts.BROSWER_MARGINS
    return f"calc({value} + {browser_margin * num_margins}px)"


## ------ Base Styles ------ ##

SIMPLE_BORDER = {"border": "1px solid black"}
STANDARD_PADDING = {
    "box-sizing": "border-box",
    "padding-top": "1vh",
    "padding-right": "1vh",
    "padding-bottom": "1vh",
    "padding-left": "1vh",
}


## ------ Node Inspector ------ ##

NI_DIV_STYLE = {
    "position": "absolute",
    "height": subtract_browser_margin("100vh", num_margins=2),
    "width": subtract_browser_margin("25vw"),
    **STANDARD_PADDING,
    **SIMPLE_BORDER,
}


COLUMN_STYLE = {"text-align": "right"}
FIRST_COLUMN_STYLE = {"text-align": "left"}
TABLE_STYLE = {"width": "100%"}

SEGMENT_HEADERS = ["Segment", "L (mm)", "Tol", "Res"]
SEGMENT_ATTRIBUTES = ["length", "tolerance", "residual_calib"]
SEGMENT_ATT_NUM_DIGITS = [0, 2, 2]
SEGMENT_SCORE_NUM_DIGITS = 2

JOINT_HEADERS = ["Joint", "Ang (deg)", "Tol", "Res"]
JOINT_ATTRIBUTES = ["angle", "tolerance", "residual_calib"]
JOINT_ATT_NUM_DIGITS = [1, 1, 1]
JOINT_SCORE_NUM_DIGITS = 1


def generate_data_entry_str(value, num_decimals: int = 1):
    if num_decimals > 3:
        LOGGER.warning(
            "Revisit rounding function for num_decimals > 3 (lazy implementation)."
        )
        return f"{value}"

    if num_decimals == 0:
        return f"{value:.0f}"
    if num_decimals == 1:
        return f"{value:.1f}"
    if num_decimals == 2:
        return f"{value:.2f}"
    if num_decimals == 3:
        return f"{value:.3f}"


def generate_table_header(labels) -> html.Thead:
    return html.Thead(
        html.Tr(
            [
                html.Th(labels[0], style=FIRST_COLUMN_STYLE),
                *[html.Th(label, style=COLUMN_STYLE) for label in labels[1:]],
            ]
        )
    )


def generate_table_rows(
    components: Union[list[Segment], list[Joint]],
    attributes: list[str],
    num_digits: list[int],
) -> list[html.Tr]:
    return [
        html.Tr(
            [
                html.Td(str(comp), style=FIRST_COLUMN_STYLE),
                *[
                    html.Td(
                        generate_data_entry_str(getattr(comp, att), nd),
                        style=COLUMN_STYLE,
                    )
                    for att, nd in zip(attributes, num_digits)
                ],
            ]
        )
        for comp in components
    ]


def generate_agg_row(score, num_digits) -> html.Tr:
    return html.Tr(
        [
            html.Th("Aggregated", style=COLUMN_STYLE),
            html.Td(""),
            html.Td(""),
            html.Td(
                generate_data_entry_str(score, num_digits),
                style={"text-align": "right", "font-weight": "bold"},
            ),
        ]
    )


def generate_table_body(
    components, attributes, num_digits, score, score_digits
) -> html.Tbody:
    rows = generate_table_rows(components, attributes, num_digits)
    rows.append(generate_agg_row(score, score_digits))

    return html.Tbody(rows)


def generate_table(
    headers, components, attributes, num_digits, score, score_digits
) -> html.Table:
    return html.Table(
        [
            generate_table_header(headers),
            generate_table_body(
                components, attributes, num_digits, score, score_digits
            ),
        ],
        style=TABLE_STYLE,
    )


def generate_ni_div():
    return html.Div(id="node-inspector", style=NI_DIV_STYLE)


def generate_ni_div_children(
    selected_node: str,
    segments: list[Segment],
    joints: list[Joint],
    component_scores: dict[str, dict],
    raw_node_scores: dict[str, float],
    node_scores: dict[str, float],
    max_score: float,
    max_marker: str,
    scoring_params: scoringParameters.ScoringParameters,
) -> list:

    segment_score = component_scores["segment"].get(selected_node, 0)
    joint_score = component_scores["joint"].get(selected_node, 0)
    raw_node_score = raw_node_scores.get(selected_node, 0)
    node_score = node_scores.get(selected_node, 0)

    threshold = scoring_params.removal_threshold.get(selected_node, 0)

    return [
        html.H2(f"Selected Node: {selected_node}"),
        html.H4("Connected Segments:"),
        html.Div(
            generate_table(
                SEGMENT_HEADERS,
                segments,
                SEGMENT_ATTRIBUTES,
                SEGMENT_ATT_NUM_DIGITS,
                segment_score,
                SEGMENT_SCORE_NUM_DIGITS,
            ),
            style={
                "height": "22vh",
                "overflow-y": "auto",
                "border": "1px solid black",
                "margin-bottom": "2vh",
            },
        ),
        html.H4("Connected Joints:"),
        html.Div(
            generate_table(
                JOINT_HEADERS,
                joints,
                JOINT_ATTRIBUTES,
                JOINT_ATT_NUM_DIGITS,
                joint_score,
                JOINT_SCORE_NUM_DIGITS,
            ),
            style={
                "height": "26vh",  # Set a fixed height for scrolling
                "overflow-y": "auto",  # Enable vertical scrolling when content exceeds height
                "border": "1px solid black",  # Border for visibility
                "margin-bottom": "2vh",  # Space below the table
            },
        ),
        html.H4(f"Raw score: {raw_node_score:.2f}  |   Threshold: {threshold:.1f}"),
        html.H3(f"Score: {node_score:.2f}"),
        html.H3(f"Max Score: {max_score:.2f} ({max_marker})"),
    ]


## ------ Interactive Graph ------ ##


IG_DIV_STYLE = {
    "position": "absolute",
    "height": "70vh",
    "width": "50vw",
    "left": "25vw",
    "box-sizing": "border-box",
    **SIMPLE_BORDER,
}


def generate_ig() -> html.Div:
    return html.Div(
        [
            dcc.Graph(id="interactive-graph", style={"height": "95%", "width": "100%"}),
            dcc.Checklist(
                id="label-toggle",
                options=[{"label": "Hide Labels", "value": "hide"}],
                value=[],
                inline=True,
                style={"height": "5%"},
            ),
        ],
        style=IG_DIV_STYLE,
    )
    # return dcc.Graph(id="interactive-graph", style=IG_DIV_STYLE)


def get_scene_limits(positions: list, slider_params) -> dict:
    scene_limits = {}
    max_range = 0
    ax_middles = {}
    for i, ax in enumerate(["x", "y", "z"]):
        min_val = min([pos[i] for pos in positions]) + slider_params["min"]
        max_val = max([pos[i] for pos in positions]) + slider_params["max"]
        if (max_val - min_val) > max_range:
            max_range = max_val - min_val
        ax_middles[ax] = (max_val + min_val) / 2

    scene_limits = {
        f"{ax}axis": {"range": [mid - max_range / 2, mid + max_range / 2]}
        for ax, mid in ax_middles.items()
    }

    return scene_limits


## ------ Node Manipulator ------ ##


NM_DIV_STYLE = {
    "position": "absolute",
    "top": add_browser_margin("70vh"),
    "left": "25vw",
    "width": "50vw",
    "height": subtract_browser_margin("30vh", num_margins=2),
    **STANDARD_PADDING,
    **SIMPLE_BORDER,
}

NM_SLIDER_DIV_STYLE = {
    "position": "absolute",
    "height": "100%",
    "width": "40%",
    "left": "0%",
}
NM_REMOVAL_DIV_STYLE = {
    "position": "absolute",
    "height": "100%",
    "width": "30%",
    "left": "40%",
}
NM_MISSING_DIV_STYLE = {
    **NM_REMOVAL_DIV_STYLE,
    "left": "70%",
}

NM_BUTTON_STYLE = {
    "position": "absolute",
    "height": "18%",
    "width": "30%",
}
NM_SLIDER_ARGS = {
    "min": -100,
    "max": 100,
    "step": 10,
    "value": 0,
    "marks": None,
    "tooltip": {"placement": "top"},
}
NM_SLIDER_STYLE = {"width": "100%", "height": "100%"}


NM_HEADER_LABEL = "Set Offsets"
NM_REMOVAL_TABLE_STYLE = {
    "height": "100%",
    "width": "100%",
    "display": "flex",
    "flex-direction": "column",
    "align-items": "flex-start",
    "overflow-y": "auto",
}
NM_TABLE_ROW_STYLE = {
    "height": "20%",
    "width": "100%",
    "display": "flex",  # Use flex to split cells evenly
    "align-items": "center",  # Align items in each row to the center vertically
    "flex-shrink": "0",  # Prevent rows from shrinking
    "min-height": "5vh",  # Set a minimum height for each row
}


def generate_nm_slider(
    name: str, slider_args, slider_style: dict = None, div_style: dict = None
) -> html.Div:
    """!Generate a slider for the Node Manipulator Panel

    @param name Name of the slider. Used for label and id ('slider-' prepended)
    @param slider_args Slider object args
    @param slider_style Style for Slider obj
    @param div_style Style used for the Div containinghte label and slider
    """
    ss = slider_style or {}
    ds = div_style or {}
    slider_id = f"slider-{'-'.join(name.lower().split(' '))}"
    return html.Div(
        [
            html.Label(f"{name}:"),
            html.Div(
                [dcc.Slider(id=slider_id, **slider_args)],
                style=slider_style,
            ),
        ],
        style=div_style,
    )


def generate_nm_slider_div_children(disabled: bool = False) -> list:
    global NM_HEADER_LABEL, NM_SLIDER_ARGS, NM_SLIDER_STYLE, NM_BUTTON_STYLE, NM_DIV_STYLE

    slider_args = {**NM_SLIDER_ARGS, "disabled": disabled}
    return [
        html.Button(
            "Reset",
            id="reset-button",
            n_clicks=0,
            style={
                "left": "65%",
                "top": "10%",
                **NM_BUTTON_STYLE,
            },
        ),
        html.Button(
            "Reset All",
            id="reset-all-button",
            n_clicks=0,
            style={"top": "40%", "left": "65%", **NM_BUTTON_STYLE},
        ),
        html.Button(
            id="remove-node-button",
            children="Remove",
            style={**NM_BUTTON_STYLE, "left": "65%", "top": "70%"},
        ),
        html.Div(
            [
                html.Label(
                    id="slider-label",
                    children=NM_HEADER_LABEL,
                ),
                generate_nm_slider("X", slider_args, NM_SLIDER_STYLE),
                generate_nm_slider("Y", slider_args, NM_SLIDER_STYLE),
                generate_nm_slider("Z", slider_args, NM_SLIDER_STYLE),
            ],
            style={
                "width": "60%",
                "left": "5%",
                "position": "absolute",
            },
        ),
    ]


def generate_nm_slider_div(disabled: bool = False) -> html.Div:
    """!Generate and return the Node Manipulator Div"""

    return html.Div(
        generate_nm_slider_div_children(disabled),
        id="nm-sliders",
        style=NM_SLIDER_DIV_STYLE,
    )


def generate_nm_removal_row(text):
    return html.Tr(
        [
            html.Td(text, style={"width": "60%"}),
            html.Td(
                html.Button(
                    "Add Back",
                    id={"type": f"add-back-button", "index": text},
                    n_clicks=0,
                    style={"width": "100%", "height": "100%"},
                ),
                style={"width": "40%", "height": "80%", "top": "10%"},
            ),
        ],
        style=NM_TABLE_ROW_STYLE,
    )


def generate_nm_removal_table(removed_nodes):
    rows = [generate_nm_removal_row(node) for node in removed_nodes]
    return html.Table(
        rows,
        style=NM_REMOVAL_TABLE_STYLE,
    )


def generate_nm_removal_div(removed_nodes=None) -> html.Div:
    removed = removed_nodes or {}
    return html.Div(
        generate_nm_removal_div_children(removed),
        id="nm-removal",
        style=NM_REMOVAL_DIV_STYLE,
    )


def generate_nm_removal_div_children(removed_nodes: dict) -> html.Div:
    return [
        html.Div(
            [
                html.Label(
                    "Removed Nodes",
                    style={
                        "left": "0%",
                        "top": "0%",
                        "position": "relative",
                    },
                ),
                generate_nm_removal_table(removed_nodes),
            ],
            style={
                "height": "80%",
                "width": "95%",
            },
        ),
    ]


def generate_nm_missing_row(text):
    return html.Tr(
        [
            html.Td(text, style={"width": "60%"}),
            html.Td(
                html.Button(
                    "Fit",
                    id={"type": f"fit-button", "index": text},
                    n_clicks=0,
                    style={"width": "100%", "height": "100%"},
                ),
                style={"width": "40%", "height": "80%", "top": "10%"},
            ),
        ],
        style=NM_TABLE_ROW_STYLE,
    )


def generate_nm_missing_table(missing_nodes):
    rows = [generate_nm_missing_row(node) for node in missing_nodes]
    return html.Table(
        rows,
        style=NM_REMOVAL_TABLE_STYLE,
    )


def generate_nm_missing_div(missing_nodes=None) -> html.Div:
    missing = missing_nodes or {}
    return html.Div(
        generate_nm_missing_div_children(missing),
        id="nm-missing",
        style=NM_MISSING_DIV_STYLE,
    )


def generate_nm_missing_div_children(missing_nodes: dict) -> html.Div:
    return [
        html.Div(
            [
                html.Label(
                    "Missing Nodes",
                    style={"top": "0%", "position": "relative"},
                ),
                generate_nm_missing_table(missing_nodes),
            ],
            style={
                "height": "80%",
                "width": "95%",
            },
        ),
    ]


def generate_nm_div(removed_nodes=None, missing_nodes=None) -> html.Div:
    """!Generate and return the Node Manipulator Div"""
    global NM_HEADER_LABEL, NM_SLIDER_ARGS, NM_SLIDER_STYLE, NM_BUTTON_STYLE, NM_DIV_STYLE

    return html.Div(
        [
            generate_nm_slider_div(),
            generate_nm_removal_div(removed_nodes),
            generate_nm_missing_div(missing_nodes),
        ],
        style=NM_DIV_STYLE,
    )


## ------ Scoring Parameter Editor ------ ##

SPE_DIV_STYLE = {
    **NI_DIV_STYLE,
    "left": "75vw",
}

SPE_BUTTON_STYLE = {
    "width": "35%",
    "height": "5vh",
}


def generate_preagg_threshold_div(
    scoring_params: scoringParameters.ScoringParameters,
) -> html.Div:
    return html.Div(
        [
            html.H4("Pre-Aggregation Thresholding", style={"margin-bottom": "2vh"}),
            html.Div(
                [
                    html.Label("Segment:"),
                    dcc.RadioItems(
                        id="preagg-segment",
                        options=[
                            {"label": "True", "value": True},
                            {"label": "False", "value": False},
                        ],
                        value=scoring_params.preagg_thresholding.segment,
                        labelStyle={
                            "display": "inline-block",
                            "margin-right": "10px",
                        },
                    ),
                ],
                style={"width": "50%", "position": "absolute"},
            ),
            html.Div(
                [
                    html.Label("Joint:"),
                    dcc.RadioItems(
                        id="preagg-joint",
                        options=[
                            {"label": "True", "value": True},
                            {"label": "False", "value": False},
                        ],
                        value=scoring_params.preagg_thresholding.joint,
                        labelStyle={
                            "display": "inline-block",
                            "margin-right": "10px",
                        },
                    ),
                ],
                style={"width": "50%", "position": "absolute", "left": "50%"},
            ),
        ],
        style={"position": "relative", "height": "10vh"},
    )


def generate_agg_method_div(
    scoring_params: scoringParameters.ScoringParameters,
) -> html.Div:
    return html.Div(
        [
            html.H4("Aggregation Method", style={"margin-bottom": "2vh"}),
            html.Div(
                [
                    html.Label("Segment:"),
                    dcc.Dropdown(
                        id="agg-method-segment",
                        options=[
                            {"label": "Mean", "value": "mean"},
                            {"label": "Sum", "value": "sum"},
                        ],
                        value=scoring_params.aggregation_method.segment,
                        clearable=False,
                        style={"width": "80%"},
                    ),
                ],
                style={"width": "50%", "position": "absolute"},
            ),
            html.Div(
                [
                    html.Label("Joint:"),
                    dcc.Dropdown(
                        id="agg-method-joint",
                        options=[
                            {"label": "Mean", "value": "mean"},
                            {"label": "Sum", "value": "sum"},
                        ],
                        value=scoring_params.aggregation_method.joint,
                        clearable=False,
                        style={"width": "80%"},
                    ),
                ],
                style={"width": "50%", "position": "absolute", "left": "50%"},
            ),
        ],
        style={"position": "relative", "height": "12vh"},
    )


def generate_agg_weight_div(
    scoring_params: scoringParameters.ScoringParameters,
) -> html.Div:

    return html.Div(
        [
            html.H4("Aggregation Weight", style={"margin-bottom": "2vh"}),
            html.Div(
                [
                    html.Label("Segment:"),
                    dcc.Input(
                        id="agg-weight-segment",
                        type="number",
                        value=scoring_params.aggregation_weight.segment,
                        style={
                            "width": "80%",
                            "text-align": "right",
                            "font-size": "1em",
                        },
                    ),
                ],
                style={"position": "absolute", "width": "50%"},
            ),
            html.Div(
                [
                    html.Label("Joint:"),
                    dcc.Input(
                        id="agg-weight-joint",
                        type="number",
                        value=scoring_params.aggregation_weight.joint,
                        style={
                            "width": "80%",
                            "text-align": "right",
                            "font-size": "1em",
                        },
                    ),
                ],
                style={"position": "absolute", "left": "50%", "width": "50%"},
            ),
        ],
        style={"position": "relative", "height": "12vh"},
    )


def generate_removal_thresh_div(
    scoring_params: scoringParameters.ScoringParameters, removal_threshold_all: float
) -> html.Div:

    dd_options = [
        {"label": "All", "value": "all"},
        *[{"label": m, "value": m} for m in scoring_params.removal_threshold.keys()],
    ]

    return html.Div(
        [
            html.H4("Removal Threshold", style={"margin-bottom": "2vh"}),
            html.Div(
                [
                    html.Label("Select Threshold:", style={"width": "50%"}),
                    dcc.Dropdown(
                        id="removal-threshold-type",
                        options=dd_options,
                        value="all",
                        clearable=False,
                        style={"width": "50%"},
                    ),
                ],
                style={
                    "position": "relative",
                    "width": "100%",
                    "display": "flex",
                    "margin-bottom": "1vh",
                },
            ),
            html.Div(
                [
                    html.Label(
                        id="removal-threshold-label",
                        children="Threshold for All:",
                        style={"width": "50%"},
                    ),
                    dcc.Input(
                        id="removal-threshold-value",
                        type="number",
                        value=removal_threshold_all,
                        style={
                            "width": "40%",
                            "text-align": "right",
                            "font-size": "1em",
                        },
                    ),
                ],
                style={"position": "relative", "width": "100%", "display": "flex"},
            ),
        ],
        style={"position": "relative", "height": "16vh"},
    )


def generate_import_export_div() -> html.Div:
    return html.Div(
        [
            dcc.Upload(
                id="import-scoring",
                children=html.Button("Import", style={**SPE_BUTTON_STYLE}),
                multiple=False,
                accept=".json",
                style={"position": "absolute", "width": "100%", "left": "10%"},
            ),
            html.Button(
                "Export",
                id="export-scoring",
                n_clicks=0,
                style={
                    **SPE_BUTTON_STYLE,
                    "position": "absolute",
                    "left": "55%",
                },
            ),
        ],
        style={
            "position": "relative",
            "width": "100%",
            "height": "5vh",
        },
    )


def generate_spe_div(
    scoring_params: scoringParameters.ScoringParameters, removal_threshold_all: float
) -> html.Div:

    return html.Div(
        [
            html.H2("Scoring Parameters"),
            generate_import_export_div(),
            generate_preagg_threshold_div(scoring_params),
            generate_agg_method_div(scoring_params),
            generate_agg_weight_div(scoring_params),
            generate_removal_thresh_div(scoring_params, removal_threshold_all),
            # Button to submit and update scoring parameters
            html.Button(
                "Update",
                id="update-button",
                n_clicks=0,
                style={**SPE_BUTTON_STYLE, "position": "relative", "left": "30%"},
            ),
            # Output for displaying updated values
            html.Div(id="output-info", style={"margin-top": "20px"}),
            dcc.Download(id="download-scoring"),
        ],
        id="sp-editor",
        style=SPE_DIV_STYLE,
    )
