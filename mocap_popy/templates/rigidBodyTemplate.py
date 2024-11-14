"""!
    Rigid Body Template
    ===================

    A template for a rigid body model.

    @author C. McCarthy
"""

import logging

from pydantic import BaseModel, Field, field_validator, model_validator

import mocap_popy.config.regex as regex
from mocap_popy.models.rigid_body import RigidBody, Node, Segment

LOGGER = logging.getLogger(__name__)


class RigidBodyTemplate(BaseModel):
    name: str
    markers: list[str] = Field(description="List of marker strings")
    segments: list[str] = Field(description="List of segment strings")
    parent_models: list[str] = Field(
        description="List of models (labeling skeleton templates) that contain this rigid body template",
        default_factory=list,
    )
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
            self.tolerances = {"segment": segment_tols, "joint": joint_tols}

        return self

    def to_rigid_body(self) -> RigidBody:
        """!Create a RigidBody instance from the RigidBodyTemplate.

        @return RigidBody instance.
        """
        nodes = [Node.from_string(marker) for marker in self.markers]
        segments = [Segment.from_string(segment) for segment in self.segments]

        return RigidBody(
            name=self.name,
            nodes=nodes,
            segments=segments,
            segment_length_tolerances=self.tolerances["segment"],
            joint_angle_tolerances=self.tolerances["joint"],
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
