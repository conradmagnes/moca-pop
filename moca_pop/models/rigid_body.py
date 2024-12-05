"""!
    Rigid Body Class and Components
    ===============================

    A rigid body is defined by a collection of Nodes, Segments, and Joints.
    Nodes are marker points in 3D space, Segments are connections between nodes, and Joints are connections between segments.

    This can be used to supplement the Vicon Nexus labeling skeleton (.vsk).
    A Rigid Body is equivalent to a segment in the VSK. Segments here are equivalent to sticks in the VSK.

    @author C. McCarthy
"""

import copy
import logging
from itertools import combinations
from typing import Union, Optional, Iterable, Literal

import matplotlib.pyplot as plt
import numpy as np
import tqdm

import moca_pop.config.regex as regex
import moca_pop.utils.string_utils as str_utils


LOGGER = logging.getLogger("RigidBody")


class Node:
    def __init__(self, marker: str, position: np.ndarray = None, exists: bool = None):
        if not regex.is_marker_string_valid(marker):
            raise ValueError(f"Invalid marker string: {marker}")

        self._marker = marker
        self._position: np.ndarray = self.validate_position(position)
        self._exists: bool = exists
        if exists is None:
            self.update_exists()

    @property
    def marker(self) -> str:
        return self._marker

    @property
    def position(self) -> np.ndarray:
        return self._position

    @position.setter
    def position(self, value: np.ndarray):
        self._position = value
        self.update_exists()

    @property
    def exists(self) -> bool:
        return self._exists

    @exists.setter
    def exists(self, value: bool):
        self._exists = value

    def get_connected_segments(
        self, segments: list["Segment"], only_existing: bool = False
    ) -> list["Segment"]:
        """!Get segments from list connected to the node.

        @param segments List of segments to search.
        """
        existing_check = lambda seg: seg.exists if only_existing else True
        return [seg for seg in segments if existing_check(seg) and self.marker in seg]

    def get_connected_joints(
        self, joints: list["Joint"], only_existing: bool = False
    ) -> list["Joint"]:
        """!Get joints from list connected to the node.

        @param joints List of joints to search.
        """
        existing_check = lambda joint: joint.exists if only_existing else True
        return [
            joint
            for joint in joints
            if existing_check(joint) and self.marker == joint.node.marker
        ]

    def __eq__(self, other: "Node") -> bool:
        """!Equality indicates that the nodes share the same marker."""
        if not isinstance(other, Node):
            return False
        return self.marker == other.marker

    def __str__(self) -> str:
        return self.marker

    def __repr__(self) -> str:
        missing = " (missing)" if not self.exists else ""
        return f"[Node] {self.marker}: pos={np.round(self.position, 2)}" + missing

    def update_exists(self) -> bool:
        """!Update the existence of the node based on its position."""
        self.exists = np.any(self.position)

    @classmethod
    def validate_position(cls, position: np.ndarray) -> np.ndarray:
        if position is None:
            return np.zeros(3)

        if len(position) != 3:
            raise ValueError(f"Invalid position array: {position}")

        if not isinstance(position, np.ndarray):
            try:
                return np.array(position)
            except ValueError:
                raise ValueError(f"Invalid position array: {position}")

        return position

    @classmethod
    def validate_nodes(
        cls,
        nodes: Iterable[Union["Node", str]],
        generate_new: bool = False,
        allow_duplicates: bool = False,
    ):
        def skip_duplicate(marker, markers, allow):
            if marker in markers:
                msg = f"Duplicate node marker: {marker}"
                if not allow:
                    raise ValueError(msg)
                else:
                    LOGGER.warning(msg)
                    return True
            return False

        input_type = type(nodes)
        valid_nodes = []
        markers = []
        for node in nodes:

            if isinstance(node, Node):
                if skip_duplicate(node.marker, markers, allow_duplicates):
                    continue

                if generate_new:
                    valid_nodes.append(Node(node.marker, node.position, node.exists))
                else:
                    valid_nodes.append(node)
                markers.append(node.marker)
            elif isinstance(node, str):
                if skip_duplicate(node, markers, allow_duplicates):
                    continue

                valid_nodes.append(Node(node))
                markers.append(node)
            else:
                raise ValueError(
                    f"Unrecognized input type for node: {type(node)}. Expected str or Node."
                )
        try:
            return input_type(valid_nodes)
        except TypeError:
            return list(valid_nodes)


class Segment:
    def __init__(
        self,
        nodes: Union[str, tuple[Union[Node, str], Union[Node, str]]],
        length: float = 0,
        tolerance: float = 0,
    ):
        """!A segment connects two nodes in 3D space.

        @param nodes Either a string in format '<marker1>-<marker2>' or a tuple of Node objects or markers.
        @param length Length of the segment.
        @param tolerance Tolerance for the segment length. (Should be expressed as a fraction of the length.)
        """
        if isinstance(nodes, str) and regex.is_segment_string_valid(nodes):
            nodes = nodes.split("-")

        if len(nodes) != 2:
            raise ValueError("Segment must be valid string or have exactly 2 nodes.")

        self._nodes = Node.validate_nodes(nodes)
        self._length = length
        self._tolerance = tolerance
        self._residual_calib = 0
        self._residual_prior = 0
        self._str = "-".join(self)

    @property
    def nodes(self) -> list[Node]:
        return self._nodes

    @property
    def length(self) -> float:
        return self._length

    @property
    def tolerance(self) -> float:
        return self._tolerance

    @tolerance.setter
    def tolerance(self, value: float):
        self._tolerance = value

    @property
    def residual_calib(self) -> float:
        return self._residual_calib

    @property
    def residual_prior(self) -> float:
        return self._residual_prior

    @property
    def exists(self) -> bool:
        """!Whether the segment exists."""
        return all([node.exists for node in self.nodes])

    def get_markers(self) -> list[str]:
        """!Get the markers connected by the segment."""
        return [node.marker for node in self.nodes]

    def get_vector(self) -> np.ndarray:
        """!Get the vector between the two nodes."""
        return self.nodes[1].position - self.nodes[0].position

    def compute_length(self):
        """!Compute and store the length of the segment."""
        if self.exists:
            self._length = np.linalg.norm(self.get_vector())
        else:
            self._length = 0

    def compute_residuals(
        self,
        reference_calib: Union["RigidBody", "Segment"] = None,
        reference_prior: Union["RigidBody", "Segment"] = None,
        reference_length: float = None,
    ):
        """!Compute the residuals of the segment length relative to the calibrated
        rigid_body and the prior rigid_body.

        Residuals are normalized to the segment length of the calibrated component, if provided.
        Otherwise, normalizes to the segment length of the prior component.

        @param reference_calib RigidBody or Segment object with calibrated segment lengths.
        @param reference_prior RigidBody or Segment object with prior segment lengths.
        @param reference_length Reference length for normalization.
        """

        self._residual_calib, ref_length = self.get_segment_residual_and_reference(
            reference_calib
        )
        ref_length = ref_length or reference_length
        self._residual_prior, _ = self.get_segment_residual_and_reference(
            reference_prior, ref_length
        )

    def get_segment_residual_and_reference(
        self,
        reference_component: Union["RigidBody", "Segment"],
        reference_length: float = None,
    ) -> tuple[float, float]:
        """!Get the residual of the segment length relative to a rigid_body and the reference length.

        Residual is normalized to the reference length, if provided.
        Otherwise, normalizes to segment length of the rigid_body.

        @param rigid_body RigidBody object to compare to.
        @return Residual and reference length.
        """

        res = 0
        ref_length = reference_length if reference_length != 0 else None

        if reference_component is None:
            return res, ref_length

        seg = (
            reference_component.get_segment(self)
            if isinstance(reference_component, RigidBody)
            else reference_component
        )
        if seg is not None and seg.exists and seg.length != 0:
            ref_length = reference_length or seg.length
            res = (self.length - seg.length) / ref_length
        return res, ref_length

    def shares_nodes(self, other: Union["Segment", tuple, str]) -> bool:
        """!Check if two segments share the same nodes.

        @param other Segment or tuple of markers.
        """
        other_seg = Segment.validate_segments([other])[0]
        return len(set(self).intersection(other_seg)) == 2

    def __iter__(self):
        """!Iterate over the markers of the segment."""
        return iter(self.get_markers())

    def __eq__(self, other: "Segment") -> bool:
        """!Equality indicates that the segments share the same nodes."""
        if not isinstance(other, Segment):
            return False
        return self.shares_nodes(other) and self.length == other.length

    def __str__(self) -> str:
        return self._str

    def __repr__(self) -> str:
        rounded_attrs = [round(attr, 2) for attr in [self.length, self.tolerance]]
        missing = " (missing)" if not self.exists else ""
        return (
            f"[Segment] {str(self)}: len={rounded_attrs[0]}, tol={rounded_attrs[1]}"
            + missing
        )

    @classmethod
    def validate_segments(
        cls,
        segments: Iterable[Union[str, tuple, "Segment"]],
        generate_new: bool = False,
    ):
        input_type = type(segments)
        valid_segments = []
        for seg in segments:
            if isinstance(seg, Segment):
                if generate_new:
                    valid_segments.append(Segment(seg.nodes))
                else:
                    valid_segments.append(seg)
            elif isinstance(seg, (tuple, str, list)):
                valid_segments.append(Segment(seg))
            else:
                raise ValueError(
                    f"Unrecognized input type for segment: {type(seg)}. Expected str, tuple, or Segment."
                )
        try:
            return input_type(valid_segments)
        except TypeError:
            return list(valid_segments)


class Joint:
    def __init__(
        self,
        segments: Union[str, tuple[Union[str, tuple, Segment]]],
        angle: float = 0,
        tolerance: float = 0,
    ):
        """!A joint connects two segments at a shared node.

        @param segments Either a string in format '<marker1>-<marker2>-<marker3>' or a tuple of Segment types.
        @param angle Angle of the joint.
        @param tolerance Tolerance for the joint angle.
        """

        if isinstance(segments, str) and regex.is_joint_string_valid(segments):
            segments = segments.split("-")

        if len(segments) == 3:
            segments = Segment.validate_segments((segments[:2], segments[1:]))
        elif len(segments) != 2:
            raise ValueError(
                "Joint must be valid string or tuple of markers or have exactly 2 segments."
            )

        segments = Segment.validate_segments(segments)
        shared_node = self.determine_shared_node(segments)
        if shared_node is None:
            raise ValueError("No common node between segments.")

        self._segments = segments
        self._node = shared_node
        self._angle = angle
        self._tolerance = tolerance
        self._residual_calib = 0
        self._residual_prior = 0

        self._str = "-".join(self)

    @property
    def segments(self):
        return self._segments

    @property
    def node(self):
        return self._node

    @property
    def angle(self):
        return self._angle

    @property
    def tolerance(self):
        return self._tolerance

    @property
    def residual_calib(self):
        return self._residual_calib

    @property
    def residual_prior(self):
        return self._residual_prior

    @property
    def exists(self):
        """!Whether the joint exists (i.e. both segments exist)."""
        return all([seg.exists for seg in self.segments])

    def get_nodes(self) -> list[Node]:
        """!Get the nodes connected to the joint. Shared node is in the middle."""
        all_nodes = [node for segment in self.segments for node in segment.nodes]
        sorted_nodes = sorted(all_nodes, key=lambda node: node.marker)
        unique = [
            i for i, node in enumerate(sorted_nodes) if node.marker != self._node.marker
        ]

        return [sorted_nodes[unique[0]], self._node, sorted_nodes[unique[1]]]

    def get_markers(self) -> list[str]:
        """!Get the markers connected to the joint."""
        return [node.marker for node in self.get_nodes()]

    def compute_angle(self):
        """!Compute the angle between two segments."""
        if not self.exists:
            self._angle = 0
            return

        nodes = self.get_nodes()
        v1 = nodes[0].position - nodes[1].position
        v2 = nodes[2].position - nodes[1].position

        norm_v1 = np.linalg.norm(v1)
        norm_v2 = np.linalg.norm(v2)

        if norm_v1 == 0 or norm_v2 == 0:
            self._angle = 0
            return

        cos_theta = np.clip(np.dot(v1, v2) / (norm_v1 * norm_v2), -1.0, 1.0)

        angle_rad = np.arccos(cos_theta)
        self._angle = np.degrees(angle_rad)

    def compute_residuals(
        self,
        reference_calib: Union["RigidBody", "Joint"] = None,
        reference_prior: Union["RigidBody", "Joint"] = None,
    ):
        """!Compute and store the residuals of the joint angle relative to its calibrated
        angle and the prior rigid_body, if provided.

        Residuals are left as absolute values.

        @param reference_calib RigidBody or Joint object with calibrated joint angles.
        @param reference_prior RigidBody or Joint object with prior joint angles.
        """
        self._residual_calib = self.get_joint_residual(reference_calib)
        self._residual_prior = self.get_joint_residual(reference_prior)

    def get_joint_residual(self, reference: Union["RigidBody", "Joint"]) -> float:
        """!Compute the residual of the joint angle relative to a rigid_body.

        @param rigid_body RigidBody object to compare to.
        """
        res = 0
        if reference is None:
            return res

        joint = (
            reference.get_joint(self.segments)
            if isinstance(reference, RigidBody)
            else reference
        )
        if joint is not None and joint.exists and joint.angle != 0:
            res = self._angle - joint.angle
        return res

    def shares_segments(self, other: Union["Joint", tuple, str]) -> bool:
        """!Check if two joints share the same segments (as indicated by their node markers).

        @param other Joint or tuple of segments.
        """
        if isinstance(other, Joint):
            joint_str = str(other)
        elif isinstance(other, tuple):
            joint_str = str(Joint(other))
        elif isinstance(other, str):
            joint_str = other
        else:
            raise ValueError(f"Unrecognized input type for comparison: {type(other)}.")

        return joint_str == str(self)

    def __iter__(self):
        """!Iterate over the markers of the joint."""
        return iter(self.get_markers())

    def __eq__(self, other) -> bool:
        """!Equality indicates that the joints share the same segments and have the same angle."""
        if not isinstance(other, Joint):
            return False
        return self.shares_segments(other) and self.angle == other.angle

    def __str__(self):
        return self._str

    def __repr__(self):
        rounded_attrs = [round(attr, 2) for attr in [self.angle, self.tolerance]]
        missing = " (missing)" if not self.exists else ""
        return (
            f"[Joint] {str(self)}: angle={rounded_attrs[0]}, tol={rounded_attrs[1]}"
            + missing
        )

    @classmethod
    def determine_shared_node(cls, segments) -> Node:
        """!Get the node shared by the segments."""
        if len(segments) != 2:
            return None
        seg1, seg2 = segments
        shared = [node for node in seg1.nodes if node in seg2.nodes]
        if len(shared) == 1:
            return shared[0]
        return None


class RigidBody:
    def __init__(
        self,
        name: str,
        nodes: Union[list[str], list[Node]],
        segments: Union[list[Segment], list[tuple], list[str]],
        segment_length_tolerances: Union[list[float], dict[str, float]] = None,
        joint_angle_tolerances: Union[list[float], dict[str, float]] = None,
        compute_segment_lengths: bool = False,
        compute_joint_angles: bool = False,
    ):
        """!A rigid_body contains a list of nodes, segments, and joints.

        Joints are inferred from the nodes and segments.

        @param name Name of the rigid_body.
        @param nodes Node markers or Node objects.
        @param segments List of segment tuples or Segment objects.
        @param segment_length_tolerances Segment length tolerances. Should be a dict with {str(segment): float} or list of floats.
        @param joint_angle_tolerance Joint angle tolerances. Angles are in degrees and set by node.
                Should be a dict with {str(node): float} or list of floats.
        @param compute_segment_lengths Whether to compute segment lengths on initialization.
        @param compute_joint_angles Whether to compute joint angles on initialization.
        """
        self._name = name
        self._nodes: list[Node] = Node.validate_nodes(nodes, generate_new=True)
        self._segments: list[Segment] = self.generate_segments(
            segments, compute_segment_lengths, segment_length_tolerances
        )
        self._joints: list[Joint] = self.generate_joints(
            self._segments, compute_joint_angles, joint_angle_tolerances
        )

    @property
    def name(self) -> str:
        return self._name

    @property
    def nodes(self) -> list[Node]:
        return self._nodes

    @property
    def segments(self) -> list[Segment]:
        return self._segments

    @property
    def joints(self) -> list[Joint]:
        return self._joints

    @property
    def percent_complete(self) -> float:
        """!Get the percentage of nodes that exist."""
        if len(self.nodes) == 0:
            return 0
        return sum([node.exists for node in self.nodes]) / len(self.nodes)

    def generate_segments(
        self,
        segments: list,
        compute_segment_lengths: bool,
        segment_length_tolerances: Union[dict, list],
    ) -> list[Segment]:
        """!Generate segments from a list of tuples or Segment objects.
        Uses own nodes to create segments, if available.

        @param segments List of segment tuples or Segment objects.
        @param compute_segment_lengths Whether to compute segment lengths on initialization.
        @param segment_length_tolerances Segment length tolerances. Either dict with {str(segment): float} or list of floats.
        """
        own_segments = []
        for i, seg in enumerate(segments):
            if isinstance(seg, Segment):
                nodes = seg.nodes
            else:
                nodes = Segment(seg).nodes

            nodes = [self.get_node(node.marker) or node for node in nodes]
            segment = Segment(nodes)

            if isinstance(segment_length_tolerances, dict):
                segment.tolerance = segment_length_tolerances.get(str(segment), 0)
            elif isinstance(segment_length_tolerances, list):
                segment.tolerance = segment_length_tolerances[i]

            if compute_segment_lengths:
                segment.compute_length()

            own_segments.append(segment)
        return own_segments

    def generate_joints(
        self,
        segments: Optional[list] = None,
        compute_angles: bool = False,
        angle_tolerances: Union[list[float], dict] = None,
    ) -> list[Joint]:
        """!Generate joints for all nodes and connected segments.

        @param segments Optional list of segments to use for joint generation. Otherwise, uses own segments.
        @param compute_angles Whether to compute joint angles on initialization.
        @param angle_tolerances Joint angle tolerances. Either dict with {str(node): float} or list of floats.
        """
        joints = []
        segments = segments or self.segments
        for i, node in enumerate(self.nodes):
            connected_segments = node.get_connected_segments(segments)
            if len(connected_segments) < 2:
                continue

            if isinstance(angle_tolerances, dict):
                tol = angle_tolerances.get(node.marker, 0)
            elif isinstance(angle_tolerances, list):
                tol = angle_tolerances[i]
            else:
                tol = 0

            for segs in combinations(connected_segments, 2):
                joint = Joint(segments=segs, tolerance=tol)
                if compute_angles:
                    joint.compute_angle()

                joints.append(joint)

        return joints

    def get_markers(self) -> list[str]:
        """!Get the markers of all nodes."""
        return [node.marker for node in self.nodes]

    def get_node(self, marker: str, exclude: list = None) -> Optional[Node]:
        """!Get a rigid_body node by marker.

        @param marker Node marker.
        @param exclude List of nodes to exclude from search.
        """
        exclude_list = exclude or []
        search_nodes = [node for node in self.nodes if node not in exclude_list]
        for node in search_nodes:
            if node.marker == marker:
                return node
        return None

    def remove_node(self, node: Union[str, Node]):
        """!Remove a node from the rigid_body (set to non-existent).

        @param node Node marker or Node object.
        """
        if isinstance(node, str):
            node = self.get_node(node)

        if node is None:
            return

        node.position = np.zeros(3)
        self.compute_segment_lengths()
        self.compute_joint_angles()

    def get_segment(
        self, markers: Union[Segment, tuple, str], exclude: list = None
    ) -> Optional[Segment]:
        """!Get a rigid_body segment by markers.

        @param markers Segment object, tuple of markers, or string of markers.
        @param exclude List of segments to exclude from search.
        """
        exclude_list = exclude or []
        search_segs = [seg for seg in self.segments if seg not in exclude_list]
        for segment in search_segs:
            if segment.shares_nodes(markers):
                return segment
        return None

    def get_connected_segments(
        self, node: Union[str, Node], only_existing: bool = False
    ) -> list[Segment]:
        """!Get segments connected to a node."""
        node = node if isinstance(node, Node) else self.get_node(node)
        if node is None:
            return []
        return node.get_connected_segments(self.segments, only_existing)

    def get_joint(
        self, segments: Union[Joint, tuple, str], exclude: list = None
    ) -> Optional[Joint]:
        """!Get a rigid_body joint by segments."""
        exclude_list = exclude or []
        search_joints = [joint for joint in self.joints if joint not in exclude_list]
        for joint in search_joints:
            if joint.shares_segments(segments):
                return joint
        return None

    def get_connected_joints(
        self, node: Union[str, Node], only_existing: bool = False
    ) -> list[Joint]:
        """!Get joints connected to a node."""
        node = node if isinstance(node, Node) else self.get_node(node)
        if node is None:
            return []
        return node.get_connected_joints(self.joints, only_existing)

    def get_shared_segments(self, rigid_body: "RigidBody") -> dict[Segment, Segment]:
        """!Get segments that are shared between two rigid_bodies.

        @param rigid_body RigidBody object to compare to.
        @return Dictionary of {self segment index: other segment} pairs.
        """
        common_segments = {}
        for segment in self.segments:
            other_seg = rigid_body.get_segment(
                segment, exclude=list(common_segments.values())
            )
            common_segments[str(segment)] = other_seg
        return common_segments

    def get_shared_joints(self, rigid_body: "RigidBody") -> dict[Joint, Joint]:
        """!Get joints that are shared between two rigid_bodies.

        @param rigid_body RigidBody object to compare to.
        @return Dictionary of {joint: joint} pairs.
        """
        common_joints = {}
        for joint in self.joints:
            other_joint = rigid_body.get_joint(
                joint, exclude=list(common_joints.values())
            )
            common_joints[str(joint)] = other_joint
        return common_joints

    def update_node_positions(
        self,
        node_positions: Union[dict[str, Union[np.ndarray, Node]], list[Node]],
        recompute_lengths: bool = False,
        recompute_angles: bool = False,
    ):
        """!Update the positions of nodes in the rigid_body. Propagates changes to segments and joints.

        @param node_positions Dictionary of {node_marker: position array} or {node_marker: Node}, or list of nodes.

        """
        if isinstance(node_positions, dict):
            for node in self.nodes:
                if node.marker in node_positions:
                    new_pos = node_positions[node.marker]
                    if isinstance(new_pos, Node):
                        new_pos = new_pos.position

                    if not isinstance(new_pos, np.ndarray):
                        LOGGER.warning("Invalid position array for node. Skipping.")
                        continue
                    node.position = new_pos

        elif isinstance(node_positions, list):
            for new_node in node_positions:
                node = self.get_node(new_node.marker)
                if node is not None:
                    node.position = new_node.position

        if recompute_lengths:
            self.compute_segment_lengths()
        if recompute_angles:
            self.compute_joint_angles()

    def get_segment_tolerances(self) -> dict[str, float]:
        """!Get the tolerances of all segments."""
        return {str(seg): seg.tolerance for seg in self.segments}

    def get_joint_tolerances(self) -> dict[str, float]:
        """!Get the tolerances of all joints."""
        return {str(joint): joint.tolerance for joint in self.joints}

    def update_segment_tolerances(self, tolerances: dict[str, float]):
        """!Update the tolerances of all segments."""
        for seg in self.segments:
            seg.tolerance = tolerances.get(str(seg), seg.tolerance)

    def update_joint_tolerances(self, tolerances: dict[str, float]):
        """!Update the tolerances of all joints."""
        for joint in self.joints:
            joint.tolerance = tolerances.get(str(joint), joint.tolerance)

    def compute_segment_lengths(self):
        """!Compute the lengths of all segments."""
        for seg in self.segments:
            seg.compute_length()

    def compute_segment_residuals(
        self,
        reference_calib: Union["RigidBody", dict[str, Segment]] = None,
        reference_prior: Union["RigidBody", dict[str, Segment]] = None,
        reference_lengths: dict[str, float] = None,
    ):
        """!Compute the residuals of all segments relative to the calibrated
        rigid_body and the prior rigid_body.

        @param reference_calib RigidBody object with calibrated segment lengths or segment mapping.
        @param reference_prior RigidBody object with prior segment lengths or segment mapping.
        @param reference_lengths Dictionary of reference segment lengths.
        """
        for seg in self.segments:
            calib = (
                reference_calib.get(str(seg), None)
                if isinstance(reference_calib, dict)
                else reference_calib
            )
            prior = (
                reference_prior.get(str(seg), None)
                if isinstance(reference_prior, dict)
                else reference_prior
            )
            ref = reference_lengths.get(str(seg), None) if reference_lengths else None
            seg.compute_residuals(calib, prior, ref)

    def compute_joint_angles(self):
        """!Compute the angles of all joints."""
        for joint in self.joints:
            joint.compute_angle()

    def compute_joint_residuals(
        self,
        reference_calib: Union["RigidBody", dict[str, Joint]] = None,
        reference_prior: Union["RigidBody", dict[str, Joint]] = None,
    ):
        """!Compute the residuals of all joints relative to the calibrated
        rigid_body and the prior rigid_body.

        @param reference_calib RigidBody object with calibrated joint angles or joint mapping.
        @param reference_prior RigidBody object with prior joint angles or joint mapping.
        """
        for joint in self.joints:
            calib = (
                reference_calib.get(str(joint), None)
                if isinstance(reference_calib, dict)
                else reference_calib
            )
            prior = (
                reference_prior.get(str(joint), None)
                if isinstance(reference_prior, dict)
                else reference_prior
            )
            joint.compute_residuals(calib, prior)

    def get_aggregate_segment_residuals(
        self,
        node: Union[str, Node],
        res_types: list[str] = None,
        average: bool = False,
        ignore_tolerance: bool = False,
    ) -> Union[float, tuple[float, float]]:
        """!Get the aggregate residuals of all segments connected to a node.

        @param node Node marker or Node object.
        @param average Whether to average the residuals. Otherwise, sum them.
        @return Tuple of the combined residuals for calibration and prior.
        """
        res_types = res_types or ["residual_calib", "residual_prior"]
        node = node if isinstance(node, Node) else self.get_node(node)
        if node is None:
            return (0, 0) if len(res_types) > 1 else 0
        connected_segments = node.get_connected_segments(
            self.segments, only_existing=True
        )

        return self.aggregate_residuals(
            connected_segments, res_types, average, ignore_tolerance
        )

    def get_aggregate_joint_residuals(
        self,
        node: Union[str, Node],
        res_types: list[str] = None,
        average: bool = False,
        ignore_tolerance: bool = False,
    ) -> Union[float, tuple[float, float]]:
        """!Get the aggregate residuals of all joints connected to a node.

        @param node Node marker or Node object.
        @param average Whether to average the residuals. Otherwise, sum them.
        @return Tuple of the combined residuals for calibration and prior.
        """
        res_types = res_types or ["residual_calib", "residual_prior"]
        node = node if isinstance(node, Node) else self.get_node(node)
        if node is None:
            return (0, 0) if len(res_types) > 1 else 0
        connected_joints = node.get_connected_joints(self.joints, only_existing=True)

        return self.aggregate_residuals(
            connected_joints, res_types, average, ignore_tolerance
        )

    @staticmethod
    def aggregate_residuals(
        components: Union[list[Joint], list[Segment]],
        res_types: list[str] = None,
        average: bool = False,
        ignore_tolerance: bool = False,
    ) -> Union[float, tuple[float, float]]:
        """!Aggregate residuals from a list of segments or joints.

        @param components List of Segment or Joint objects.
        @param average Whether to average the residuals. Otherwise, sum them.
        @param ignore_tolerance Whether to ignore the tolerance when computing residuals.
        """
        res_types = res_types or ["residual_calib", "residual_prior"]
        agg_residuals = []
        for res_type in res_types:
            residuals = []
            for comp in components:
                tol = 0 if ignore_tolerance else comp.tolerance
                res = max(0, abs(getattr(comp, res_type)) - tol)
                residuals.append(res)

            agg_res = (
                sum(residuals) / max(1, len(residuals)) if average else sum(residuals)
            )
            agg_residuals.append(agg_res)

        return tuple(agg_residuals) if len(agg_residuals) > 1 else agg_residuals[0]

    def draw_nodes_on(
        self,
        ax: plt.Axes,
        plane: str,
        include_labels: bool = False,
        color: str = "b",
        text_kwargs: dict = None,
    ):
        """!Draw nodes on a plot axis.

        @param ax Matplotlib axis to draw on.
        @param plane Plane to draw nodes on (i.e. 'xy', 'xyz').
        @param include_labels Whether to include node labels.
        @param color Color of the nodes.
        @param text_kwargs Additional text parameters.
        """
        ax_map = {"x": 0, "y": 1, "z": 2}
        default_label_kwargs = dict(fontsize=12, ha="right")
        text_kwargs = {**default_label_kwargs, **(text_kwargs or {})}

        for node in self.nodes:
            pos = node.position
            coord = [pos[ax_map[p]] for p in plane]
            ax.scatter(*coord, c=color)
            if include_labels:
                ax.text(*coord, node.marker, **text_kwargs)

        return

    def draw_segments_on(self, ax: plt.Axes, plane: str, color: str = "b"):
        """!Draw segments on a plot axis.

        @param ax Matplotlib axis to draw on.
        @param plane Plane to draw segments on (i.e. 'xy', 'xyz').
        @param color Color of the segments.
        """
        ax_map = {"x": 0, "y": 1, "z": 2}
        for segment in self.segments:
            start = segment.nodes[0].position
            end = segment.nodes[1].position
            start_coord = [start[ax_map[p]] for p in plane]
            end_coord = [end[ax_map[p]] for p in plane]
            ax.plot(*zip(start_coord, end_coord), c=color)

        return

    def __str__(self):
        return self.name

    def __repr__(self):
        node_break = str_utils.get_section_break_str("Nodes")
        segment_break = str_utils.get_section_break_str("Segments")
        joint_break = str_utils.get_section_break_str("Joints")

        perc_complete = round(self.percent_complete * 100)
        name_str = f"({perc_complete}%) [RigidBody] {self.name}"
        node_str = "\n".join([repr(node) for node in self.nodes])
        segments_str = "\n".join([repr(seg) for seg in self.segments])
        joint_str = "\n".join([repr(joint) for joint in self.joints])
        return "".join(
            [
                name_str,
                node_break,
                node_str,
                segment_break,
                segments_str,
                joint_break,
                joint_str,
            ]
        )


def best_fit_transform(target_body: RigidBody, active_body: RigidBody) -> bool:
    """
    Computes the optimal rigid transformation (rotation + translation) that aligns
    points A with points B using the Kabsch algorithm.

    @param target_body Rigid body to be transformed.
    @param active_body Rigid body to be used as the reference.
    @return Whether the transformation was successful.
    """
    active_nodes = [n for n in active_body.nodes if n.exists]
    if len(active_nodes) < 3:
        return False

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

    return True


def apply_transform(points, R, t):
    """
    Applies the given rigid transformation (rotation + translation) to the points.

    @param points (N, 3) ndarray of points to transform.
    @param R (3, 3) ndarray representing the rotation matrix.
    @param t (3, ) ndarray representing the translation vector.

    @return transformed_points (N, 3) ndarray of transformed points.
    """
    return (R @ points.T).T + t
