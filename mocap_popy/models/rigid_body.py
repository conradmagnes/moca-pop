"""!
    Rigid Body Class and Components
    ===============================

    A rigid body is defined by a collection of Nodes, Segments, and Joints.
    Nodes are marker points in 3D space, Segments are connections between nodes, and Joints are connections between segments.

    This can be used to supplement the Vicon Nexus labeling skeleton (.vsk).
    A Rigid Body is equivalent to a segment in the VSK. Segments here are equivalent to sticks in the VSK.

    @author C. McCarthy
"""

import dataclasses
from itertools import combinations
from typing import Union, Optional
import logging

import numpy as np

import mocap_popy.config.logger as logger

LOGGER = logger.setup_logger(__name__)


@dataclasses.dataclass
class Node:
    marker: str
    position: np.ndarray
    exists: bool

    def get_connected_segments(self, segments: list["Segment"]) -> list["Segment"]:
        """!Get segments from list connected to the node.

        @param segments List of segments to search.
        """
        return [
            seg for seg in segments if seg.exists and self.marker in seg.get_markers()
        ]

    def get_connected_joints(self, joints: list["Joint"]) -> list["Joint"]:
        """!Get joints from list connected to the node.

        @param joints List of joints to search.
        """
        return [
            joint
            for joint in joints
            if joint.exists and self.marker == joint.node.marker
        ]

    def __str__(self) -> str:
        return f"{self.marker}: {np.round(self.position, 2)}"


@dataclasses.dataclass
class Segment:
    node1: Node
    node2: Node
    length: float = 0
    tolerance: float = 0
    residual_calib: float = 0
    residual_prior: float = 0

    @property
    def exists(self) -> bool:
        """!Whether the segment exists."""
        return self.node1.exists and self.node2.exists

    def get_nodes(self) -> list[Node]:
        """!Get the nodes connected by the segment."""
        return [self.node1, self.node2]

    def get_markers(self) -> list[str]:
        """!Get the markers connected by the segment."""
        return [self.node1.marker, self.node2.marker]

    def get_vector(self) -> np.ndarray:
        """!Get the vector between the two nodes."""
        return self.node2.position - self.node1.position

    def compute_length(self):
        """!Compute and store the length of the segment."""
        if self.exists:
            self.length = np.linalg.norm(self.get_vector())
        else:
            self.length = 0

    def compute_residuals(
        self, rigid_body_calib: "RigidBody" = None, rigid_body_prior: "RigidBody" = None
    ):
        """!Compute the residuals of the segment length relative to the calibrated
        rigid_body and the prior rigid_body.

        Residuals are normalized to the segment length of the calibrated rigid_body, if provided.
        Otherwise, normalizes to the segment length of the prior rigid_body.

        @param rigid_body_calib RigidBody object with calibrated segment lengths.
        @param rigid_body_prior RigidBody object with prior segment lengths.
        """

        self.residual_calib, ref_length = self.get_segment_residual_and_reference(
            rigid_body_calib
        )
        self.residual_prior, _ = self.get_segment_residual_and_reference(
            rigid_body_prior, ref_length
        )

    def get_segment_residual_and_reference(
        self, rigid_body: "RigidBody", reference_length: float = None
    ) -> tuple[float, float]:
        """!Get the residual of the segment length relative to a rigid_body and the reference length.

        Residual is normalized to the reference length, if provided.
        Otherwise, normalizes to segment length of the rigid_body.

        @param rigid_body RigidBody object to compare to.
        @return Residual and reference length.
        """

        res = 0
        ref_length = reference_length if reference_length != 0 else None

        if rigid_body is None:
            return res, ref_length

        seg = rigid_body.get_segment(*self.get_markers())
        if seg is not None and seg.exists and seg.length != 0:
            ref_length = reference_length or seg.length
            res = (self.length - seg.length) / ref_length
        return res, ref_length

    def shares_nodes(self, other: Union["Segment", tuple]) -> bool:
        """!Check if two segments share the same nodes.

        @param other Segment or tuple of markers.
        """

        if isinstance(other, tuple):
            other_markers = [t.marker if isinstance(t, Node) else t for t in other]
        elif isinstance(other, Segment):
            other_markers = other
        else:
            raise ValueError(f"Unrecognized input type for comparison: {type(other)}.")
        return len(set(self).intersection(other_markers)) == 2

    def __iter__(self):
        """!Iterate over the markers of the segment."""
        return iter(self.get_markers())

    def __eq__(self, other: "Segment") -> bool:
        """!Equality indicates that the segments share the same nodes."""
        if not isinstance(other, Segment):
            return False
        return self.shares_nodes(other) and self.length == other.length

    def __str__(self) -> str:
        return f"{self.node1.marker}-{self.node2.marker}: {round(self.length, 2)}"


@dataclasses.dataclass
class Joint:
    node: Node
    segment1: Segment
    segment2: Segment
    angle: float = 0
    tolerance: float = 0
    residual_calib: float = 0
    residual_prior: float = 0

    def __post_init__(self):
        if (
            self.node not in self.segment1.get_nodes()
            or self.node not in self.segment2.get_nodes()
        ):
            raise ValueError("Node not connected to both segments.")

    @property
    def exists(self):
        """!Whether the joint exists (i.e. both segments exist)."""
        return self.segment1.exists and self.segment2.exists

    def get_nodes(self) -> list[Node]:
        """!Get the nodes connected to the joint. Shared node is in the middle."""
        second_node = [node for node in self.segment1.get_nodes() if node != self.node][
            0
        ]
        third_node = [node for node in self.segment2.get_nodes() if node != self.node][
            0
        ]

        return [second_node, self.node, third_node]

    def get_markers(self) -> list[str]:
        """!Get the markers connected to the joint."""
        return [node.marker for node in self.get_nodes()]

    def get_segments(self) -> list[Segment]:
        """!Get the segments connected to the joint."""
        return [self.segment1, self.segment2]

    def compute_angle(self):
        """!Compute the angle between two segments."""
        if not self.exists:
            self.angle = 0
            return

        nodes = self.get_nodes()
        v1 = nodes[0].position - nodes[1].position
        v2 = nodes[2].position - nodes[1].position

        norm_v1 = np.linalg.norm(v1)
        norm_v2 = np.linalg.norm(v2)

        if norm_v1 == 0 or norm_v2 == 0:
            self.angle = 0
            return

        cos_theta = np.clip(np.dot(v1, v2) / (norm_v1 * norm_v2), -1.0, 1.0)

        angle_rad = np.arccos(cos_theta)
        self.angle = np.degrees(angle_rad)

    def compute_residuals(
        self, rigid_body_calib: "RigidBody" = None, rigid_body_prior: "RigidBody" = None
    ):
        """!Compute and store the residuals of the joint angle relative to its calibrated
        angle and the prior rigid_body, if provided.

        Residuals are left as absolute values.

        @param rigid_body_calib RigidBody object with calibrated joint angles.
        @param rigid_body_prior RigidBody object with prior joint angles.
        """
        self.residual_calib = self.get_joint_residual(rigid_body_calib)
        self.residual_prior = self.get_joint_residual(rigid_body_prior)

    def get_joint_residual(self, rigid_body: "RigidBody"):
        """!Compute the residual of the joint angle relative to a rigid_body.

        @param rigid_body RigidBody object to compare to.
        """
        res = 0
        if rigid_body is None:
            return res

        joint = rigid_body.get_joint(self.segment1, self.segment2)
        if joint is not None and joint.exists and joint.angle != 0:
            res = self.angle - joint.angle
        return res

    def shares_segments(self, other: Union["Joint", tuple]) -> bool:
        """!Check if two joints share the same segments (as indicated by their node markers).

        @param other Joint or tuple of segments.
        """
        if isinstance(other, tuple):
            segments: list[Segment] = []
            for ot in other:
                try:
                    if isinstance(ot, Segment):
                        segments.append(ot)
                    else:
                        segments.append(Segment(ot[0], ot[1]))
                except IndexError:
                    raise ValueError(
                        f"Unrecognized input type for comparison: {tuple([type(ot) for ot in other])}."
                    )

            other_joint = Joint(self.node, segments[0], segments[1])
        else:
            other_joint = other

        return len(set(self).intersection(other_joint)) == 3

    def __iter__(self):
        """!Iterate over the markers of the joint."""
        return iter(self.get_markers())

    def __eq__(self, other) -> bool:
        """!Equality indicates that the joints share the same node and segments."""
        if not isinstance(other, Joint):
            return False
        return self.node == other.node and self.shares_segments(other)

    def __str__(self):
        return f"{'-'.join(self)}: {round(self.angle, 2)}"


class RigidBody:
    def __init__(
        self,
        nodes: Union[list[str], list[Node]],
        segments: Union[list[Segment], list[tuple]],
        segment_length_tolerances: list[float] = None,
        joint_angle_tolerances: list[float] = None,
        compute_segment_lengths: bool = False,
        compute_joint_angles: bool = False,
    ):
        """!A rigid_body contains a list of nodes, segments, and joints.

        Joints are inferred from the nodes and segments.

        @param nodes Node markers or Node objects.
        @param segments List of segment tuples or Segment objects.
        @param segment_length_tolerances Segment length tolerances. The list should match the order of segments.
        @param joint_angle_tolerance Joint angle tolerances. Angles are in degrees and set by node. The list should match the order of nodes.
        @param compute_segment_lengths Whether to compute segment lengths on initialization.
        @param compute_joint_angles Whether to compute joint angles on initialization.
        """
        self.nodes: list[Node] = []
        self.segments: list[Segment] = []
        self.joints: list[Joint] = []

        for node in nodes:
            if isinstance(node, str):
                self.nodes.append(Node(node, np.zeros(3), False))
            else:
                self.nodes.append(Node(node.marker, node.position, node.exists))

        for i, seg in enumerate(segments):
            tol = segment_length_tolerances[i] if segment_length_tolerances else 0
            if isinstance(seg, tuple):
                node1 = self.get_node(seg[0])
                node2 = self.get_node(seg[1])
                if node1 is None or node2 is None:
                    LOGGER.debug(f"Node missing for segment {seg}")
                    continue
                segment = Segment(node1, node2, tolerance=tol)
            else:
                segment = Segment(seg.node1, seg.node2, tolerance=tol)

            if compute_segment_lengths:
                segment.compute_length()

            self.segments.append(segment)

        self._generate_joints(compute_joint_angles, joint_angle_tolerances)

    def _generate_joints(
        self, compute_angles: bool = False, angle_tolerances: list[float] = None
    ):
        """!Generate joints for all nodes and connected segments.
        This is called during initialization and should not be called directly.
        """
        self.joints = []
        for i, node in enumerate(self.nodes):
            connected_segments = node.get_connected_segments(self.segments)
            if len(connected_segments) < 2:
                continue

            tol = angle_tolerances[i] if angle_tolerances else 0
            for seg1, seg2 in combinations(connected_segments, 2):
                joint = Joint(node=node, segment1=seg1, segment2=seg2, tolerance=tol)
                if compute_angles:
                    joint.compute_angle()
                self.joints.append(joint)

    def get_node(self, marker: str) -> Optional[Node]:
        """!Get a rigid_body node by marker."""
        for node in self.nodes:
            if node.marker == marker:
                return node
        return None

    def get_segment(self, marker1: str, marker2: str) -> Optional[Segment]:
        """!Get a rigid_body segment by markers."""
        for segment in self.segments:
            if segment.shares_nodes((marker1, marker2)):
                return segment
        return None

    def get_connected_segments(self, node: Union[str, Node]) -> list[Segment]:
        """!Get segments connected to a node."""
        node = node if isinstance(node, Node) else self.get_node(node)
        if node is None:
            return []
        return node.get_connected_segments(self.segments)

    def get_joint(
        self, segment1: Union[Segment, tuple[str]], segment2: Union[Segment, tuple[str]]
    ) -> Optional[Joint]:
        """!Get a rigid_body joint by segments."""
        for joint in self.joints:
            if joint.shares_segments((segment1, segment2)):
                return joint
        return None

    def get_connected_joints(self, node: Union[str, Node]) -> list[Joint]:
        """!Get joints connected to a node."""
        node = node if isinstance(node, Node) else self.get_node(node)
        if node is None:
            return []
        return node.get_connected_joints(self.joints)

    def compute_segment_lengths(self):
        """!Compute the lengths of all segments."""
        for seg in self.segments:
            seg.compute_length()

    def compute_segment_residuals(
        self,
        rigid_body_calib: "RigidBody" = None,
        rigid_body_prior: "RigidBody" = None,
    ):
        """!Compute the residuals of all segments relative to the calibrated
        rigid_body and the prior rigid_body.

        @param rigid_body_calib RigidBody object with calibrated segment lengths.
        @param rigid_body_prior RigidBody object with prior segment lengths.
        """
        for seg in self.segments:
            seg.compute_residuals(rigid_body_calib, rigid_body_prior)

    def compute_joint_angles(self):
        """!Compute the angles of all joints."""
        for joint in self.joints:
            joint.compute_angle()

    def compute_joint_residuals(
        self,
        rigid_body_calib: "RigidBody" = None,
        rigid_body_prior: "RigidBody" = None,
    ):
        """!Compute the residuals of all joints relative to the calibrated
        rigid_body and the prior rigid_body.

        @param rigid_body_calib RigidBody object with calibrated joint angles.
        @param rigid_body_prior RigidBody object with prior joint angles.
        """
        for joint in self.joints:
            joint.compute_residuals(rigid_body_calib, rigid_body_prior)

    def get_aggregate_segment_residuals(
        self,
        node: Union[str, Node],
        average: bool = False,
        ignore_tolerance: bool = False,
    ) -> tuple[float, float]:
        """!Get the aggregate residuals of all segments connected to a node.

        @param node Node marker or Node object.
        @param average Whether to average the residuals. Otherwise, sum them.
        @return Tuple of the combined residuals for calibration and prior.
        """
        node = node if isinstance(node, Node) else self.get_node(node)
        if node is None:
            return 0, 0
        connected_segments = node.get_connected_segments(self.segments)

        return self.aggregate_residuals(connected_segments, average, ignore_tolerance)

    def get_aggregate_joint_residual(
        self,
        node: Union[str, Node],
        average: bool = False,
        ignore_tolerance: bool = False,
    ) -> tuple[float, float]:
        """!Get the aggregate residuals of all joints connected to a node.

        @param node Node marker or Node object.
        @param average Whether to average the residuals. Otherwise, sum them.
        @return Tuple of the combined residuals for calibration and prior.
        """
        node = node if isinstance(node, Node) else self.get_node(node)
        if node is None:
            return 0, 0
        connected_joints = node.get_connected_joints(self.joints)

        return self.aggregate_residuals(connected_joints, average, ignore_tolerance)

    @staticmethod
    def aggregate_residuals(
        components: Union[list[Joint], list[Segment]],
        average: bool = False,
        ignore_tolerance: bool = False,
    ):
        """!Aggregate residuals from a list of segments or joints.

        @param components List of Segment or Joint objects.
        @param average Whether to average the residuals. Otherwise, sum them.
        @param ignore_tolerance Whether to ignore the tolerance when computing residuals.
        """
        agg_residuals = []
        for res_type in ["residual_calib", "residual_prior"]:
            residuals = []
            for comp in components:
                tol = 0 if ignore_tolerance else comp.tolerance
                res = max(0, abs(getattr(comp, res_type)) - tol)
                residuals.append(res)

            agg_res = (
                sum(residuals) / max(1, len(residuals)) if average else sum(residuals)
            )
            agg_residuals.append(agg_res)

        return tuple(agg_residuals)

    def __str__(self):
        node_str = "\n".join([node.__str__() for node in self.nodes])
        segments_str = "\n".join([seg.__str__() for seg in self.segments])
        joint_str = "\n".join([joint.__str__() for joint in self.joints])
        return f"Nodes:\n{node_str}\nSegments:\n{segments_str}\nJoints:\n{joint_str}"
