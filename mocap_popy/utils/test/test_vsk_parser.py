import os
import re
import unittest

import mocap_popy.utils.vsk_parser as vsk_parser

THIS_PATH = os.path.join(os.path.dirname(__file__))
TEST_VSK_FILE = "test_nushu_wholefoot.vsk"
TEST_VSK_FP = os.path.join(THIS_PATH, "data", TEST_VSK_FILE)

MODEL = "NUSHU_shoe_10_wholefoot"
SEGMENTS = ["Right_Foot", "Left_Foot"]
MARKERS = [
    "Right_HH",
    "Right_HB",
    "Right_HL",
    "Right_ML",
    "Right_TL",
    "Right_TF",
    "Right_TM",
    "Right_MM",
    "Right_HM",
    "Right_TH",
    "Left_HH",
    "Left_HB",
    "Left_HL",
    "Left_ML",
    "Left_TL",
    "Left_TF",
    "Left_TM",
    "Left_MM",
    "Left_HM",
    "Left_TH",
]

TEST_PARAMETERS = {
    "Right_Foot_Right_Foot1_x": 0.918,
    "Right_Foot_Right_Foot1_y": -3.961,
    "Right_Foot_Right_Foot1_z": 0.293,
    "Left_Foot_Left_Foot2_x": 0.123,
    "Left_Foot_Left_Foot2_y": -3.456,
    "Left_Foot_Left_Foot2_z": 0.789,
}


class TestVskParser(unittest.TestCase):
    def test_parse_vsk(self):
        model, segments, parameters, markers, sticks = vsk_parser.parse_vsk(TEST_VSK_FP)

        self.assertEqual(model, MODEL)
        self.assertEqual(segments, SEGMENTS)
        self.assertEqual(markers, MARKERS)

    def test_parse_marker_positions(self):
        marker_positions = vsk_parser.parse_marker_positions_by_segment(TEST_PARAMETERS)

        expected_res = {
            "Right_Foot": {1: (0.918, -3.961, 0.293)},
            "Left_Foot": {2: (0.123, -3.456, 0.789)},
        }
        self.assertEqual(marker_positions, expected_res)

    def test_format_marker_positions(self):
        marker_positions = vsk_parser.parse_marker_positions_by_segment(TEST_PARAMETERS)

        formatted_by_list = vsk_parser.format_marker_positions(
            marker_positions, ["R1", "L2"]
        )

        formatted_by_dict = vsk_parser.format_marker_positions(
            marker_positions, {"Right_Foot": {1: "R1"}, "Left_Foot": {2: "L2"}}
        )

        expected_res = {
            "R1": (0.918, -3.961, 0.293),
            "L2": (0.123, -3.456, 0.789),
        }
        self.assertEqual(formatted_by_list, expected_res)
        self.assertEqual(formatted_by_dict, expected_res)


if __name__ == "__main__":

    unittest.main()
