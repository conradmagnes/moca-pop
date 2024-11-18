"""!
    Vicon Skeleton (VSK) File Parser
    ================================

    This module provides functions to parse Vicon Skeleton (VSK) files.

    @author C. McCarthy
"""

import re
from typing import Union
import xml.etree.ElementTree as ET

from mocap_popy.config.regex import SEGMENT_PARAM_MARKER_RGX


def parse_vsk(file_path: str) -> tuple:
    """!Parse a Vicon Skeleton (VSK) File

    The following information is parsed and returned:
    - kinematic model name
    - list of segment names
    - dict of segment parameters
    - list of marker names
    - list of sticks, or marker pairs (marker 1, marker 2)

    @param file_path path to the VSK file
    """

    tree = ET.parse(file_path)
    root = tree.getroot()

    model = root.attrib.get("MODEL")

    segments = []
    skeleton = root.find("Skeleton")
    for param in skeleton.findall(".//Segment"):
        segments.append(param.attrib.get("NAME"))

    parameters = {}
    for param in root.find("Parameters"):
        name = param.attrib.get("NAME")
        value = param.attrib.get("VALUE")
        parameters[name] = float(value)

    markers = []
    for marker in root.find("MarkerSet/Markers"):
        markers.append(marker.attrib.get("NAME"))

    sticks = []
    for stick in root.find("MarkerSet/Sticks"):
        marker1 = stick.attrib.get("MARKER1")
        marker2 = stick.attrib.get("MARKER2")
        sticks.append((marker1, marker2))

    return model, segments, parameters, markers, sticks


def parse_marker_positions_by_segment(
    parameters: dict,
    parameter_rgx: str = None,
) -> dict:
    """!Parse Marker Positions from Segment Parameters

    @param parameters dict of segment parameters
    @param parameter_rgx regular expression for segment parameter names

    @return dict of marker positions in format {segment: {marker_index: (x, y, z)}}
    """
    param_rgx = parameter_rgx or SEGMENT_PARAM_MARKER_RGX

    marker_positions = {}
    for param, value in parameters.items():
        match = re.match(param_rgx, param)
        if not match:
            continue

        segment_name = match.group("segment_name")
        idx = match.group("idx")
        axis = match.group("axis")
        if segment_name not in marker_positions:
            marker_positions[segment_name] = {}

        if idx not in marker_positions[segment_name]:
            marker_positions[segment_name][idx] = {}

        marker_positions[segment_name][idx][axis] = value

    for segment in marker_positions.values():
        for idx, pos_dict in segment.items():
            segment[idx] = tuple(pos_dict[axis] for axis in "xyz")

    return marker_positions


def format_marker_positions(
    marker_positions: dict, marker_mapping: Union[dict, list]
) -> dict:
    """!Format Marker Positions

    @param marker_positions dict of marker positions in format {segment: {marker_index: (x, y, z)}}
    @param marker_mapping Either dict specifying marker index to name for each segment or list of marker names
            In the later case, asssumes list of marker names is ordered by segment

    @return dict of marker positions in format {marker_name: (x, y, z)}
    """
    formatted = {}
    i = 0
    for segment, markers in marker_positions.items():
        for idx, pos in markers.items():
            if isinstance(marker_mapping, dict):
                marker_name = marker_mapping[segment][idx]
            else:
                marker_name = marker_mapping[i]

            formatted[marker_name] = pos
            i += 1

    return formatted
