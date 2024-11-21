"""!
    Rigid Body Template
    ===================

    A template for a rigid body model.

    @author C. McCarthy
"""

import logging

from pydantic import BaseModel, Field, field_validator, model_validator

import mocap_popy.config.regex as regex
from mocap_popy.model_templates.baseTemplate import BaseTemplate
from mocap_popy.models.rigid_body import RigidBody

LOGGER = logging.getLogger(__name__)


class RigidBodyTemplate(BaseTemplate):
    markers: list[str] = Field(description="List of marker strings")
    segments: list[str] = Field(description="List of segment strings")
    param_index_marker_mapping: dict[str, str] = Field(
        description="Mapping of index used in segment parameters to marker name",
        default_factory=dict,
    )
    tolerances: dict[str, dict[str, float]] = Field(
        description="Tolerances for segment lengths and joint angles. Joint angles tols are defined for each node.",
        default_factory=dict,
    )

    @field_validator("markers", mode="before")
    def validate_markers(cls, markers):
        if not isinstance(markers, list):
            raise ValueError("Markers must be a list of strings.")
        for marker in markers:
            if not regex.is_marker_string_valid(marker):
                raise ValueError(f"Invalid marker: {marker}")
        return markers

    @field_validator("segments", mode="before")
    def validate_segments(cls, segments):
        if not isinstance(segments, list):
            raise ValueError("Segments must be a list of strings.")
        for segment in segments:
            if not regex.is_segment_string_valid(segment):
                raise ValueError(f"Invalid segment: {segment}")
        return segments

    @model_validator(mode="after")
    def set_defaults(self):
        if not self.param_index_marker_mapping:
            self.param_index_marker_mapping = {
                str(idx): marker for idx, marker in enumerate(self.markers, start=1)
            }

        if not self.tolerances:
            segment_tols = {segment: 0 for segment in self.segments}
            joint_tols = {marker: 0 for marker in self.markers}
            self.tolerances = {"segments": segment_tols, "joints": joint_tols}

        return self

    def add_symmetry_prefix(self, side: str):
        """!Add a symmetry prefix to the markers and segments in the template."""
        self.markers = [f"{side}_{marker}" for marker in self.markers]

        sided_segments = []
        for segment in self.segments:
            seg_parts = segment.split("-")
            seg_parts = [f"{side}_{part}" for part in seg_parts]
            segment = "-".join(seg_parts)
            sided_segments.append(segment)
        self.segments = sided_segments

        for idx, marker in self.param_index_marker_mapping.items():
            self.param_index_marker_mapping[idx] = f"{side}_{marker}"

        seg_tolerance = {}
        for seg_name, seg_tol in self.tolerances["segments"].items():
            seg_name_parts = seg_name.split("-")
            seg_name_parts = [f"{side}_{part}" for part in seg_name_parts]
            sided_seg_name = "-".join(seg_name_parts)
            seg_tolerance[sided_seg_name] = seg_tol

        self.tolerances["segments"] = seg_tolerance

        joint_tolerance = {}
        for joint_name, joint_tol in self.tolerances["joints"].items():
            sided_joint_name = f"{side}_{joint_name}"
            joint_tolerance[sided_joint_name] = joint_tol

        self.tolerances["joints"] = joint_tolerance

    def convert_vsk_marker_positions(self, marker_positions: dict) -> dict:
        """!Convert marker positions from VSK format to template format.

        @param marker_positions dict of marker positions in format {marker_index: arr(x, y, z)}
        @return dict of marker positions in format {marker_name: arr(x, y, z)}
        """
        formatted = {}
        for idx, pos in marker_positions.items():
            marker_name = self.param_index_marker_mapping.get(idx, None)
            if marker_name is None:
                LOGGER.warning(f"Marker index {idx} not found in template {self.name}.")
                continue
            formatted[marker_name] = pos

        return formatted

    def to_rigid_body(self, nodes: list = None) -> RigidBody:
        """!Create a RigidBody instance from the RigidBodyTemplate.

        @return RigidBody instance.
        """
        compute_lengths = nodes is not None
        compute_joints = nodes is not None
        nodes = nodes or self.markers

        return RigidBody(
            name=self.name,
            nodes=nodes,
            segments=self.segments,
            segment_length_tolerances=self.tolerances["segments"],
            joint_angle_tolerances=self.tolerances["joints"],
            compute_segment_lengths=compute_lengths,
            compute_joint_angles=compute_joints,
        )

    @classmethod
    def from_rigid_body(cls, rigid_body: RigidBody):
        """!Create a RigidBodyTemplate from a RigidBody instance.

        @param rigid_body RigidBody instance.
        """
        markers = [str(n) for n in rigid_body.nodes]
        segments = [str(seg) for seg in rigid_body.segments]

        return RigidBodyTemplate(
            name=rigid_body.name, markers=markers, segments=segments
        )
