import unittest

from mocap_popy.models.rigid_body import RigidBody, Node, Segment
from mocap_popy.model_templates.rigidBodyTemplate import RigidBodyTemplate
from mocap_popy.config import regex


class TestRigidBodyTemplate(unittest.TestCase):
    def setUp(self):
        """Set up reusable data for tests."""
        self.valid_markers = ["Marker1", "Marker2", "Marker3"]
        self.valid_segments = ["Marker1-Marker2", "Marker2-Marker3"]

    def test_markers_validation(self):
        """Test that marker validation works."""
        template = RigidBodyTemplate(
            name="RigidBody1", markers=self.valid_markers, segments=self.valid_segments
        )
        self.assertEqual(template.markers, self.valid_markers)

        # Test invalid markers
        with self.assertRaises(ValueError):
            RigidBodyTemplate(
                name="RigidBody1",
                markers=["Invalid-Marker"],
                segments=self.valid_segments,
            )

    def test_segments_validation(self):
        """Test that segment validation works."""
        template = RigidBodyTemplate(
            name="RigidBody1", markers=self.valid_markers, segments=self.valid_segments
        )
        self.assertEqual(template.segments, self.valid_segments)

        # Test invalid segments
        with self.assertRaises(ValueError):
            RigidBodyTemplate(
                name="RigidBody1",
                markers=self.valid_markers,
                segments=["Invalid-Segment-Namee"],
            )

    def test_default_param_index_marker_mapping(self):
        """Test that param_index_marker_mapping is generated correctly."""
        template = RigidBodyTemplate(
            name="RigidBody1", markers=self.valid_markers, segments=self.valid_segments
        )
        expected_mapping = {"1": "Marker1", "2": "Marker2", "3": "Marker3"}
        self.assertEqual(template.param_index_marker_mapping, expected_mapping)

    def test_default_tolerances(self):
        """Test that tolerances are set correctly."""
        template = RigidBodyTemplate(
            name="RigidBody1", markers=self.valid_markers, segments=self.valid_segments
        )
        expected_tolerances = {
            "segments": {"Marker1-Marker2": 0, "Marker2-Marker3": 0},
            "joints": {"Marker1": 0, "Marker2": 0, "Marker3": 0},
        }
        self.assertEqual(template.tolerances, expected_tolerances)

    def test_add_symmetry_prefix(self):
        """Test adding a symmetry prefix to the template."""
        template = RigidBodyTemplate(
            name="RigidBody1", markers=self.valid_markers, segments=self.valid_segments
        )
        template.add_symmetry_prefix("Left")
        self.assertEqual(template.name, "Left_RigidBody1")
        self.assertEqual(
            template.markers, ["Left_Marker1", "Left_Marker2", "Left_Marker3"]
        )
        self.assertEqual(
            template.segments,
            ["Left_Marker1-Left_Marker2", "Left_Marker2-Left_Marker3"],
        )
        self.assertEqual(
            template.param_index_marker_mapping,
            {"1": "Left_Marker1", "2": "Left_Marker2", "3": "Left_Marker3"},
        )
        self.assertEqual(
            template.tolerances["segments"],
            {"Left_Marker1-Left_Marker2": 0, "Left_Marker2-Left_Marker3": 0},
        )
        self.assertEqual(
            template.tolerances["joints"],
            {"Left_Marker1": 0, "Left_Marker2": 0, "Left_Marker3": 0},
        )

    def test_to_rigid_body(self):
        """Test conversion to a RigidBody instance."""
        template = RigidBodyTemplate(
            name="RigidBody1", markers=self.valid_markers, segments=self.valid_segments
        )
        rigid_body = template.to_rigid_body()

        self.assertEqual(rigid_body.name, "RigidBody1")
        self.assertEqual([str(node) for node in rigid_body.nodes], self.valid_markers)
        self.assertEqual(
            [str(segment) for segment in rigid_body.segments], self.valid_segments
        )
        self.assertEqual(rigid_body.get_segment("Marker1-Marker2").tolerance, 0)
        self.assertEqual(rigid_body.get_joint("Marker1-Marker2-Marker3").tolerance, 0)

    def test_from_rigid_body(self):
        """Test creation of a template from a RigidBody instance."""
        # Mock RigidBody with nodes and segments
        rigid_body = RigidBody(
            name="RigidBody1",
            nodes=self.valid_markers,
            segments=self.valid_segments,
            segment_length_tolerances={"Marker1-Marker2": 0, "Marker2-Marker3": 0},
            joint_angle_tolerances={"Marker1": 0, "Marker2": 0, "Marker3": 0},
        )

        template = RigidBodyTemplate.from_rigid_body(rigid_body)
        self.assertEqual(template.name, "RigidBody1")
        self.assertEqual(template.markers, self.valid_markers)
        self.assertEqual(template.segments, self.valid_segments)

    def test_empty_defaults(self):
        """Test behavior when optional fields are omitted."""
        template = RigidBodyTemplate(
            name="RigidBody1", markers=self.valid_markers, segments=self.valid_segments
        )
        self.assertEqual(template.parent_models, [])
        self.assertEqual(
            template.param_index_marker_mapping,
            {"1": "Marker1", "2": "Marker2", "3": "Marker3"},
        )
        self.assertEqual(
            template.tolerances["segments"],
            {"Marker1-Marker2": 0, "Marker2-Marker3": 0},
        )
        self.assertEqual(
            template.tolerances["joints"], {"Marker1": 0, "Marker2": 0, "Marker3": 0}
        )


if __name__ == "__main__":
    unittest.main()
