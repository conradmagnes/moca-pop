import os
import unittest

import matplotlib.pyplot as plt
import numpy as np

import mocap_popy.utils.c3d_parser as c3d_parser
import mocap_popy.utils.plot_utils as plotUtils


THIS_PATH = os.path.abspath(os.path.dirname(__file__))
TEST_DATA_PATH = os.path.join(THIS_PATH, "data", "trial01.c3d")
TEST_PLT_DIR = os.path.join(THIS_PATH, "plts")

EXPECTED_POINT_LABELS = [
    "Right_HH",
    "Right_HB",
    "Right_ML",
    "Right_TL",
    "Right_TF",
    "Right_TH",
    "Left_HH",
    "Left_HB",
    "Left_HL",
    "Left_ML",
    "Left_TL",
    "Left_TF",
    "Left_TH",
    "Right_HL",
    "Right_TM",
    "Left_TM",
    "Left_HM",
    "Left_MM",
    "Right_MM",
    "Right_HM",
    "*20",
]

PLOT_MARKER = "Right_HM"


class TestC3DLoader(unittest.TestCase):
    def test_point_consistency(self):
        reader = c3d_parser.get_reader(TEST_DATA_PATH)
        self.assertTrue(c3d_parser.check_point_consistency(reader) is None)

    def test_point_labels(self):
        reader = c3d_parser.get_reader(TEST_DATA_PATH)
        point_labels = c3d_parser.get_point_labels(reader)
        self.assertEqual(point_labels, EXPECTED_POINT_LABELS)

    def test_marker_positions(self):
        reader = c3d_parser.get_reader(TEST_DATA_PATH)
        positions, residuals = c3d_parser.get_marker_positions(reader)
        self.assertTrue(len(positions) > 0)
        self.assertTrue(len(residuals) > 0)

        desired_pos, _ = c3d_parser.get_marker_positions(
            reader, marker_labels=["Right_HM"]
        )
        self.assertEqual(len(desired_pos), 1)

    def test_marker_trajectories(self):
        reader = c3d_parser.get_reader(TEST_DATA_PATH)
        trajectories = c3d_parser.get_marker_trajectories(reader)
        self.assertTrue(len(trajectories) > 0)

        desired_traj = c3d_parser.get_marker_trajectories(
            reader, marker_labels=["Right_HM"]
        )
        self.assertEqual(len(desired_traj), 1)

    def test_plot(self):
        reader = c3d_parser.get_reader(TEST_DATA_PATH)
        trajectories = c3d_parser.get_marker_trajectories(
            reader, marker_labels=[PLOT_MARKER], unit_scale=0.001
        )
        positions, residuals = c3d_parser.get_marker_positions(
            reader, marker_labels=[PLOT_MARKER], unit_scale=0.001
        )

        fig, ax = plt.subplots(1, 2, figsize=(10, 4), sharex=True, sharey=True)

        ax[0].plot(trajectories[PLOT_MARKER].x, label="x")
        ax[0].plot(trajectories[PLOT_MARKER].y, label="y")
        ax[0].plot(trajectories[PLOT_MARKER].z, label="z")

        non_existent = np.array(trajectories[PLOT_MARKER].exists) == False
        exists_l, exists_g = plotUtils.get_binary_signal_edges(non_existent)
        plotUtils.draw_shaded_regions(
            ax[0], exists_l, exists_g, color="gray", alpha=0.5, label="non-existent"
        )

        pos_arr = np.array(positions[PLOT_MARKER])
        ax[1].plot(pos_arr)

        res_binarr = np.array(residuals[PLOT_MARKER]) != 0
        res_l, res_g = plotUtils.get_binary_signal_edges(res_binarr)
        plotUtils.draw_shaded_regions(
            ax[1], res_l, res_g, color="gray", alpha=0.5, label="residual != 0"
        )

        ax[0].set_title(f"{PLOT_MARKER} Trajectories")
        ax[1].set_title(f"{PLOT_MARKER} Positions/Residuals")

        for i in range(2):
            ax[i].set_xlabel("Frame")
            ax[i].set_ylabel("Position (m)")
            ax[i].legend()

        fig.tight_layout()
        trial_name = os.path.basename(TEST_DATA_PATH).split(".")[0]
        fig.savefig(os.path.join(TEST_PLT_DIR, f"{trial_name}_{PLOT_MARKER}.png"))


if __name__ == "__main__":
    unittest.main()
