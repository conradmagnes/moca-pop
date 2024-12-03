"""!
    NUSHU Pipeline Runner
    ===============

    Runs through a series of pipelines in Vicon Nexus to process motion capture data.

    Currently hard-coded pipeline sequence for processing ETH_NUSHU trial data, consisting of the following steps:
        1) Reconstruct and Label (ETH_NUSHU_R&L)
        2) Unlabel Rigid Body Markers and Delete Unlabeled Trajectories (ETH_NUSHU_UnlabelRB)
        3) Recursive Gap Fill (ETH_NUSHU_FillGaps_Pass 1-3)
            i) Small gap fill with Woltering, Rigid Body, and Pattern Fill
            ii) Medium - Large gap fill with Kinematic Gap Fill and Rigid Body Fill
            iii) Fill remaining gaps with Kinematic Gap Fill
        4) Butterworth Filter (ETH_NUSHU_Filter)

    @author: C. McCarthy
"""

import argparse
import logging
import os
import time
import sys

from viconnexusapi import ViconNexus

import mocap_popy.config.logger as logger
from mocap_popy.utils import vicon_quality_check as vqc

LOGGER = logging.getLogger("PipelineRunner")


def run_pipeline(vicon: ViconNexus.ViconNexus, pipeline_args):
    LOGGER.info(f"Running pipeline: {pipeline_args[0]}")
    start = time.time()
    vicon.RunPipeline(*pipeline_args)
    end = time.time()
    LOGGER.info(f"Pipeline {pipeline_args[0]} completed in {end - start:.2f} seconds.")


def recursive_gap_fill(
    vicon, subject_name, pass_number=1, max_passes=3, timeout: int = 200
):
    pipeline_name = f"ETH_NUSHU_FillGaps_Pass{pass_number}"
    pipeline_args = (pipeline_name, "Shared", timeout)

    run_pipeline(vicon, pipeline_args)

    marker_trajectories = vqc.get_marker_trajectories(vicon, subject_name)
    gaps = vqc.get_num_gaps(marker_trajectories)
    labeled = vqc.get_perc_labeled(marker_trajectories)

    vqc.log_gaps(gaps)
    vqc.log_labeled(labeled)

    if sum(gaps.values()) == 0 or pass_number >= max_passes:
        return

    recursive_gap_fill(vicon, subject_name, pass_number + 1, max_passes, timeout)


def configure_parser():
    """!Configure the argument parser."""
    parser = argparse.ArgumentParser(
        description="Unassign rigid body markers based on residual scores."
    )

    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Increase logging verbosity to debug.",
    )

    parser.add_argument(
        "-l",
        "--log",
        action="store_true",
        help="Log output to file.",
    )
    parser.add_argument(
        "-s",
        "--save",
        action="store_true",
        help="Save trial upon completion.",
    )
    parser.add_argument(
        "-pp",
        "--project_path",
        type=str,
        default="",
        help="Path to the project directory. If none, uses active trial in Vicon Nexus.",
    )
    parser.add_argument(
        "-tn",
        "--trial_name",
        type=str,
        default="",
        help="Name of the trial to process. If none, uses active trial in Vicon Nexus.",
    )
    parser.add_argument(
        "-sn",
        "--subject_name",
        type=str,
        default="",
        help="Name of the subject to process. If none, uses first subject in the trial.",
    )

    return parser


def validate_file_paths(args, vicon):
    """!Validate the project path and trial name.

    @param args Argument parser arguments
    @param vicon ViconNexus instance

    @return project_path, trial_name, trial_path
    """
    if args.project_path:
        if not os.path.exists(args.project_path):
            LOGGER.error(f"Project path does not exist: {args.project_path}")
            exit(-1)

        project_path = args.project_path
        trial_name = None
    else:
        project_path, trial_name = vicon.GetTrialName()

    if not args.trial_name and not trial_name:
        _, trial_name = vicon.GetTrialName()
    elif args.trial_name:
        trial_name = args.trial_name

    trial_path = os.path.join(project_path, trial_name)
    if not os.path.exists(trial_path + ".system"):
        LOGGER.error(f"Trial not found: {trial_path}")
        exit(-1)

    return project_path, trial_name, trial_path


def main():

    parser = configure_parser()
    args = parser.parse_args()

    if args.verbose:
        LOGGER.setLevel(logging.DEBUG)

    vicon = ViconNexus.ViconNexus()
    if not vicon.IsConnected():
        LOGGER.error("Vicon Nexus is not connected.")
        exit(-1)

    project_path, trial_name, trial_path = validate_file_paths(args, vicon)

    mode = "w" if args.log else "off"
    if mode == "w":
        log_path = os.path.join(project_path, "logs")
        os.makedirs(log_path, exist_ok=True)
        logger.set_log_dir(log_path)

    logger.set_root_logger(name=f"{trial_name}_pipeline-runner", mode=mode)

    LOGGER.info(f"Launching Pipeline Runner for trial: {trial_name}")
    LOGGER.debug(f"Arguments: {args}")

    start = time.time()

    _, is_open = vicon.GetTrialName()
    if not is_open:
        LOGGER.info(f"Opening trial: {trial_path}")
        vicon.OpenTrial(trial_path, 200)

    if not args.subject_name:
        subject_name = vicon.GetSubjectNames()[0]
    elif args.subject_name not in vicon.GetSubjectNames():
        LOGGER.error(f"Subject not found: {args.subject_name}")
        exit(-1)
    else:
        subject_name = args.subject_name

    LOGGER.info(f"Subject: {subject_name}")

    run_pipeline(vicon, ("ETH_NUSHU_R&L", "Shared", 200))
    run_pipeline(vicon, ("ETH_NUSHU_UnlabelRB", "Shared", 200))

    marker_trajectories = vqc.get_marker_trajectories(vicon, subject_name)
    gaps = vqc.get_num_gaps(marker_trajectories)
    labeled = vqc.get_perc_labeled(marker_trajectories)

    vqc.log_gaps(gaps)
    vqc.log_labeled(labeled)

    if sum(gaps.values()) > 0:
        recursive_gap_fill(vicon, subject_name)

    run_pipeline(vicon, ("ETH_NUSHU_Filter", "Shared", 200))

    if args.save:
        LOGGER.info("Saving trial.")
        vicon.SaveTrial(30)

    LOGGER.info(
        "Pipeline complete. Total duration: {:.2f} seconds.".format(time.time() - start)
    )


def test_main_with_args():
    sys.argv = [
        "pipeline_runner.py",
        "-v",
        "-l",
        "-pp",
        "D:\HPL\pipeline_test_2",
        "-tn",
        "20241107T102745Z_semitandem-r",
        "-sn",
        "subject",
    ]
    main()


if __name__ == "__main__":
    test_main_with_args()
    # main()
