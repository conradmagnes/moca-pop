"""!
    Pipeline Runner
    =====================

    Runs through a series of pipelines in Vicon Nexus to process motion capture data.
    This script must be run as a standalone script. It cannot be run as a pipeline step in Vicon Nexus.

    The script extracts pipelines from a configuration file. By default, the script looks for a configuration file
    in the `config` directory of this script's parent. 
    See `pipeline.py` for details on pipeline and pipeline series configurations.


    By default, the trial will not be saved after the runner finishes (unless explicitly set to do so by a pipeline step).
    Saving the trial is not recommended as it can overwrite the original data (e.g. if filtering the data).
    Exporting the data to C3D is recommended instead.

    Some common operations (e.g. filtering, exporting) are added as flags to the script. This allows the user to
    easily add these steps to the series with a gate check for the flag.

    Usage:
    ------

    python pipeline_runner.py -off -cn <config_name> -pn <project_name> -tn <trial_name> -sn <subject_name>

    Example:
        python pipeline_runner.py -v -l -cn "nushu_pipeline_series" -pn "D:\HPL\pipeline_test_2" -tn "20241107T102745Z_semitandem-r" -sn "subject" --keep_trial_open

    Options:
    --------
    Run 'python pipeline_runner.py -h' for options.

    Returns:
    --------
    0 if successful, -1 if failed.

    @author: C. McCarthy
"""

import argparse
import logging
import os
import time
import sys

from viconnexusapi import ViconNexus
import pydantic

from moca_pop.config import logger, directory

from moca_pop.utils import json_utils, vicon_utils, quality_check as qc
import moca_pop.aux_scripts.pipeline_runner.pipeline as pipeline

LOGGER = logging.getLogger("PipelineRunner")


def validate_config_name(config_name):
    """!Validate the configuration name.

    @param config_name Path to the configuration file (or name if in default location)

    @return config_path
    """
    if not config_name:
        LOGGER.error("Configuration name not provided.")
        exit(-1)

    if os.path.exists(config_name):
        config_path = os.path.normpath(config_name)
    else:
        config_dir = os.path.join(directory.AUX_DIR, "pipeline_runner", "config")
        cn = config_name.split(".")[0]
        config_path = os.path.join(config_dir, f"{cn}.json")
        if not os.path.exists(config_path):
            LOGGER.error(f"Configuration file not found: {config_name}")
            exit(-1)

    return config_path


def validate_file_paths(vicon, project_name, trial_name):
    """!Validate the project name and trial name.

    @param config_name Path to the configuration file (or name if in default location)
    @param project_name Path to the project directory
    @param trial_name Name of the trial to process
    @param vicon ViconNexus instance

    @return project_dir, trial_name, trial_path
    """
    if project_name:
        if not os.path.exists(project_name):
            LOGGER.error(f"Path to the project {project_name} does not exist.")
            exit(-1)

        project_dir = os.path.normpath(project_name)
        valid_trial_name = None
    else:
        project_dir, valid_trial_name = vicon.GetTrialName()

    if not trial_name and not valid_trial_name:
        _, valid_trial_name = vicon.GetTrialName()
    elif trial_name:
        valid_trial_name = trial_name

    trial_path = os.path.join(project_dir, valid_trial_name)
    if not os.path.exists(trial_path + ".system"):
        LOGGER.error(f"Trial not found: {trial_path}")
        exit(-1)

    return project_dir, valid_trial_name, trial_path


def configure_parser():
    """!Configure the argument parser."""
    parser = argparse.ArgumentParser(
        description="Run a series of pipelines in Vicon Nexus to process motion capture data."
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
        "-cn",
        "--config_name",
        type=str,
        required=True,
        help="Name of the pipeline series configuration.",
    )
    parser.add_argument(
        "-pn",
        "--project_name",
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
    parser.add_argument(
        "--export",
        action="store_true",
        help="Export processed results as C3D.",
    )
    parser.add_argument(
        "--filter",
        action="store_true",
        help="Apply a Butterworth filter to the trajectories.",
    )
    parser.add_argument(
        "--keep_trial_open",
        action="store_true",
        help="Keep the trial open after processing.",
    )
    parser.add_argument(
        "--hide_quality_check",
        action="store_true",
        help="Hide the quality check output for gap fill pipelines.",
    )
    parser.add_argument(
        "--_new_console_opened",
        action="store_true",
        help=argparse.SUPPRESS,
    )

    return parser


def main():

    parser = configure_parser()
    args = parser.parse_args()

    if args.verbose:
        LOGGER.setLevel(logging.DEBUG)

    vicon = ViconNexus.ViconNexus()
    if not vicon.IsConnected():
        LOGGER.error("Vicon Nexus is not connected.")
        exit(-1)

    config_path = validate_config_name(args.config_name)
    project_dir, trial_name, trial_path = validate_file_paths(
        vicon, args.project_name, args.trial_name
    )

    config_json_str = json_utils.import_json_as_str(config_path)
    try:
        pipeline_series = pipeline.PipelineSeries.model_validate_json(config_json_str)
    except pydantic.ValidationError as e:
        LOGGER.error(f"Invalid pipeline configuration: {e}")
        exit(-1)

    mode = "w" if args.log else "off"
    if mode == "w":
        log_path = os.path.join(project_dir, "logs")
        os.makedirs(log_path, exist_ok=True)
        logger.set_log_dir(log_path)

    logger.set_root_logger(name=f"{trial_name}_pipeline-runner", mode=mode)

    LOGGER.info(f"Launching Pipeline Runner for trial: {trial_name}")
    LOGGER.debug(f"Arguments: {args}")

    start = time.time()

    _, is_open = vicon.GetTrialName()
    if not is_open or is_open != trial_name:
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

    gate_checks = {"export": args.export, "filter": args.filter}
    pipeline_series.run(
        vicon,
        gate_checks=gate_checks,
        subject_name=subject_name,
        log_quality=(not args.hide_quality_check),
    )

    if not args.hide_quality_check:
        LOGGER.info("Final quality report")
        marker_trajectories = vicon_utils.get_marker_trajectories(vicon, subject_name)
        gaps = qc.get_num_gaps(marker_trajectories)
        labeled = qc.get_perc_labeled(marker_trajectories)
        qc.log_gaps(gaps)
        qc.log_labeled(labeled)

    if not (args.keep_trial_open or is_open):
        LOGGER.info("Closing trial.")
        vicon.CloseTrial(30)

    LOGGER.info(
        "Pipeline complete. Total duration: {:.2f} seconds.".format(time.time() - start)
    )
    exit(0)


def test_main_with_args():
    sys.argv = [
        "pipeline_runner.py",
        "-v",
        "-l",
        "-cn",
        "nushu_pipeline_series.json",
        # "-pp",
        # "D:\HPL\pipeline_test_2",
        # "-tn",
        # "20241107T102745Z_semitandem-r",
        # "-sn",
        # "subject",
    ]
    main()


if __name__ == "__main__":
    # test_main_with_args()
    main()
