import unittest
import re
from moca_pop.config.regex import (
    MARKER_RGX,
    SEGMENT_RGX,
    JOINT_RGX,
    SEGMENT_PARAM_SEGMENT_RGX,
    is_marker_string_valid,
    is_segment_string_valid,
    is_joint_string_valid,
    parse_symmetrical_component,
    same_side,
)


class TestRigidBodyRegex(unittest.TestCase):
    def test_marker_string_valid(self):
        # Valid markers
        self.assertTrue(is_marker_string_valid("Marker1"))
        self.assertTrue(is_marker_string_valid("Marker_123"))
        self.assertTrue(is_marker_string_valid("Marker.Name"))

        # Invalid markers
        self.assertFalse(is_marker_string_valid("Marker-Name"))  # Contains "-"
        self.assertFalse(is_marker_string_valid("Marker,Name"))  # Contains ","
        self.assertFalse(is_marker_string_valid(""))  # Empty string

    def test_segment_string_valid(self):
        # Valid segments
        self.assertTrue(is_segment_string_valid("Marker1-Marker2"))
        self.assertTrue(is_segment_string_valid("A-B"))

        # Invalid segments
        self.assertFalse(is_segment_string_valid("Marker1-Marker1"))
        # Duplicate markers
        self.assertFalse(is_segment_string_valid("Marker1,"))  # Invalid characters
        self.assertFalse(is_segment_string_valid("Marker1-Marker2-Marker3"))
        # Too many markers
        self.assertFalse(is_segment_string_valid("Marker1-"))  # Incomplete segment

    def test_joint_string_valid(self):
        # Valid joints
        self.assertTrue(is_joint_string_valid("Marker1-Marker2-Marker3"))
        self.assertTrue(is_joint_string_valid("A-B-C"))

        # Invalid joints
        # Duplicate markers
        self.assertFalse(is_joint_string_valid("Marker1-Marker2-Marker1"))
        # Missing a third marker
        self.assertFalse(is_joint_string_valid("Marker1-Marker2"))
        # Incomplete joint
        self.assertFalse(is_joint_string_valid("Marker1-Marker2-"))
        # Invalid characters
        self.assertFalse(is_joint_string_valid("Marker1-Marker2-Marker3,"))

    def test_segment_param_segment_regex(self):
        # Valid VSK segment parameters
        self.assertTrue(re.match(SEGMENT_PARAM_SEGMENT_RGX, "Segment_Segment1_x"))
        self.assertTrue(re.match(SEGMENT_PARAM_SEGMENT_RGX, "Arm_Arm1_y"))
        self.assertTrue(re.match(SEGMENT_PARAM_SEGMENT_RGX, "Leg_Leg2_z"))

        # Invalid VSK segment parameters
        # Mismatched segment names
        self.assertFalse(re.match(SEGMENT_PARAM_SEGMENT_RGX, "Segment1_Segment1_x"))
        # Missing index and axis
        self.assertFalse(re.match(SEGMENT_PARAM_SEGMENT_RGX, "Segment_Segment"))
        # Missing axis
        self.assertFalse(re.match(SEGMENT_PARAM_SEGMENT_RGX, "Segment_Segment1"))
        # Missing axis
        self.assertFalse(re.match(SEGMENT_PARAM_SEGMENT_RGX, "Segment1_Segment1_1"))
        # Invalid axis
        self.assertFalse(re.match(SEGMENT_PARAM_SEGMENT_RGX, "Segment_Segment1_1_q"))

    def test_symmetrical_component(self):
        # Valid symmetrical components
        self.assertEqual(
            parse_symmetrical_component("Left_Component"), ("Left", "Component", "_")
        )
        self.assertEqual(
            parse_symmetrical_component("Right_Component"), ("Right", "Component", "_")
        )
        self.assertEqual(
            parse_symmetrical_component("R_Component"), ("R", "Component", "_")
        )
        self.assertEqual(
            parse_symmetrical_component("Right_Arm"), ("Right", "Arm", "_")
        )

        self.assertEqual(
            parse_symmetrical_component("Component_Left"), ("Left", "Component", "_")
        )

        self.assertEqual(parse_symmetrical_component("RTMarker"), ("R", "Marker", "T"))

        # Invalid symmetrical components
        # Missing side
        with self.assertRaises(ValueError):
            parse_symmetrical_component("Component")
        # Invalid side
        with self.assertRaises(ValueError):
            parse_symmetrical_component("Middle_Component")
        # Missing component
        with self.assertRaises(ValueError):
            parse_symmetrical_component("Left_")
        # Invalid separator
        with self.assertRaises(ValueError):
            parse_symmetrical_component("Left-Component")

    def test_same_side(self):
        # Valid same side components

        self.assertTrue(same_side("Left", "Left"))
        self.assertTrue(same_side("Right", "Right"))
        self.assertTrue(same_side("L", "left"))
        self.assertTrue(same_side("Right", "right"))


if __name__ == "__main__":
    unittest.main()
