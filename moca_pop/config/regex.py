"""!
    Regular expressions for the project.
    ================================
"""

import re

"""
Side regex, can be either: ['Right', 'Left', 'right', 'left', 'R', 'L']
"""
SIDE_RGX = r"^(Right|Left|right|left|R|L)$"
LSIDE_RGX = r"^(Left|left|L)$"
RSIDE_RGX = r"^(Right|right|R)$"

"""
Marker can contain any character except for: ['-', ',']
"""
MARKER_RGX = r"^[^-,]+$"

"""
(Rigid Body) Segments have the string format:
    <Marker1>-<Marker2>

Where <Marker1> and <Marker2> are valid and unique markers.
"""
SEGMENT_RGX = rf"^(?P<marker1>{MARKER_RGX[1:-1]})-(?P<marker2>{MARKER_RGX[1:-1]})$"


"""
(Rigid Body) Joints have the string format:
    <Marker1>-<Marker2>-<Marker3>

Where <Marker1>, <Marker2>, and <Marker3> are valid and unique markers.
"""
JOINT_RGX = rf"^(?P<marker1>{MARKER_RGX[1:-1]})-(?P<marker2>{MARKER_RGX[1:-1]})-(?P<marker3>{MARKER_RGX[1:-1]})$"


"""
Rigid Body can contain any character except for: ['-', ',']
"""
RIGID_BODY_RGX = r"^[^-,]+$"

"""
Bilateral symmetry constraints have the string format:
    <side><sep><some component>
    <some component><sep><side>

Sep can be either: ['_', 'T']
"""
SYMMETRY_PATTERN = re.compile(r"^(Right|Left|right|left|R|L)(_|T)(.+)$")
SYMMETRY_PATTERN_2 = re.compile(r"^(.+)(_|T)(Right|Left|right|left|R|L)$")

"""
(VSK) Segment parameters in VSK files can be in the format:
    <SegmentName>_<SegmentName><Index>_<Axis>
"""
SEGMENT_PARAM_SEGMENT_RGX = (
    r"^(?P<segment_name>.+)_(?P=segment_name)(?P<idx>\d+)_(?P<axis>[xyz])$"
)

"""
(VSK) They can also be in the following format:
    <SegmentName>_<MarkerName>_<Axis>
"""
SEGMENT_PARAM_MARKER_RGX = r"^(?P<segment_name>.+)_(?P<marker_name>.+)_(?P<axis>[xyz])$"


def is_marker_string_valid(marker: str) -> bool:
    """!Check if a marker string is valid.

    Markers can contain any character except for: ['-', ',']
    """
    return bool(re.match(MARKER_RGX, marker))


def is_segment_string_valid(segment: str) -> bool:
    """!Check if a (rigid body) segment string is valid.

    Segments have the string format:
        <Marker1>-<Marker2>
    where <Marker1> and <Marker2> are valid and unique markers.
    """
    match = re.match(SEGMENT_RGX, segment)
    if not match:
        return False

    marker1 = match.group("marker1")
    marker2 = match.group("marker2")

    return marker1 != marker2


def is_joint_string_valid(joint: str) -> bool:
    """!Check if a (rigid body) joint string is valid.

    Joints have the string format:
        <Marker1>-<Marker2>-<Marker3>
    where <Marker1>, <Marker2>, and <Marker3> are valid and unique markers.
    """
    match = re.match(JOINT_RGX, joint)
    if not match:
        return False

    marker1 = match.group("marker1")
    marker2 = match.group("marker2")
    marker3 = match.group("marker3")

    return marker1 != marker2 and marker1 != marker3 and marker2 != marker3


def parse_symmetrical_component(component: str) -> tuple[str, str, str]:
    """!Parse a symmetrical component string.

    Symmetrical components have the string format:
        <Side>_<component> or <component>_<Side>

    @returns side, component, sep
    """
    match = re.match(SYMMETRY_PATTERN, component)
    if match:
        return match.group(1), match.group(3), match.group(2)

    match = re.match(SYMMETRY_PATTERN_2, component)
    if match:
        return match.group(3), match.group(1), match.group(2)

    raise ValueError(f"Invalid symmetry component: {component}")


def same_side(sym1: str, sym2: str) -> bool:
    """!Compare two symmetry denotations"""

    if re.match(LSIDE_RGX, sym1) and re.match(LSIDE_RGX, sym2):
        return True
    if re.match(RSIDE_RGX, sym1) and re.match(RSIDE_RGX, sym2):
        return True
    return False
