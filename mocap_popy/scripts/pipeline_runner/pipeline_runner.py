"""!
    Pipeline Runner
    ===============

    author: C. McCarthy
"""

import logging
import os
import time

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


def main():

    project_path = "D:\HPL\pipeline_test_2"
    trial_name = "20241107T102745Z_semitandem-r"
    trial_path = os.path.join(project_path, trial_name)

    start = time.time()
    LOGGER.info(f"Opening trial: {trial_path}")

    vicon = ViconNexus.ViconNexus()
    vicon.OpenTrial(trial_path, 200)

    subject_name = vicon.GetSubjectNames()[0]

    LOGGER.info(f"Executing pipeline for subject: {subject_name}")

    run_pipeline(vicon, ("ETH_NUSHU_R&L", "Shared", 200))
    run_pipeline(vicon, ("ETH_NUSHU_UnlabelRB", "Shared", 200))

    marker_trajectories = vqc.get_marker_trajectories(vicon, subject_name)
    gaps = vqc.get_num_gaps(marker_trajectories)
    labeled = vqc.get_perc_labeled(marker_trajectories)

    vqc.log_gaps(gaps)
    vqc.log_labeled(labeled)

    if sum(gaps.values()) > 0:
        recursive_gap_fill(vicon, subject_name)

    # run_pipeline(vicon, ("ETH_NUSHU_Filter", "Shared", 200))

    LOGGER.info(
        "Pipeline complete. Total duration: {:.2f} seconds.".format(time.time() - start)
    )


if __name__ == "__main__":
    main()
