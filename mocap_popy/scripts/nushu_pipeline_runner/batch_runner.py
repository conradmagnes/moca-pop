"""!
    Batch Runner
    ============

    This script runs the NUSHU pipeline runner on a batch of trials. It is intended to be run as a standalone script.

    @author: C. McCarthy
"""

import argparse
import logging
import os
import sys
import time

from mocap_popy.config import logger, directory
from mocap_popy.utils import dist_utils
import mocap_popy.scripts.nushu_pipeline_runner.nushu_pipeline_runner as npr

LOGGER = logging.getLogger("BatchRunner")


def configure_parser():
    """!Configure the argument parser."""
    parser = argparse.ArgumentParser(
        description="Run the NUSHU pipeline runner on a batch of trials."
    )

    parser.add_argument(
        "-l",
        "--log",
        action="store_true",
        help="Enable logging to file (propogates to pipeline runners).",
    )

    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose logging (propogates to pipeline runners).",
    )

    parser.add_argument(
        "-pp",
        "--project_path",
        type=str,
        required=True,
        help="Path to the project directory",
    )

    parser.add_argument(
        "--exclude",
        type=str,
        help="Comma-separated list of trials to exclude",
    )

    parser.add_argument(
        "-sn",
        "--subject-name",
        type=str,
        help="Name of the subject",
    )

    parser.add_argument(
        "--filter",
        action="store_true",
        help="Add filter pipeline to runners.",
    )

    return parser


def main():

    parser = configure_parser()
    args = parser.parse_args()

    if args.verbose:
        LOGGER.setLevel(logging.DEBUG)

    project_path = args.project_path
    if not os.path.isdir(project_path):
        LOGGER.error(f"Invalid project path: {project_path}")
        exit(-1)

    mode = "w" if args.log else "off"
    if mode == "w":
        log_path = os.path.join(project_path, "logs")
        os.makedirs(log_path, exist_ok=True)
        logger.set_log_dir(log_path)

    logger.set_root_logger(name=f"batch-runner", mode=mode)

    LOGGER.info("Batch Runner started.")
    start = time.time()
    LOGGER.debug(f"Arguments: {args}")

    trials = [
        f.split(".")[0] for f in os.listdir(project_path) if f.endswith(".Trial.enf")
    ]
    excluded_trials = args.exclude.split(",") if args.exclude else []
    trials = [t for t in trials if t not in excluded_trials]

    LOGGER.info(f"Trials: {', '.join(trials)}")

    pipeline_runner_path = os.path.join(
        directory.SCRIPTS_DIR, "nushu_pipeline_runner", "nushu_pipeline_runner.py"
    )

    common_args = ["-e", "-pp", project_path]
    if args.verbose:
        common_args.append("-v")
    if args.log:
        common_args.append("-l")
    if args.subject_name:
        common_args.extend(["-sn", args.subject_name])
    if args.filter:
        common_args.append("--filter")

    for trial in trials:
        trial_start = time.time()
        LOGGER.info(f"Running pipeline for trial: {trial}")
        script_args = [*common_args, "-tn", trial]
        process = dist_utils.run_python_script(
            pipeline_runner_path,
            script_args,
            # verbose=args.verbose,
            python_path=sys.executable,
        )
        if process.returncode != 0:
            LOGGER.error(f"Pipeline Runner failed for trial: {trial}")
            break

        LOGGER.info(
            "Trial runner completed in {:.1f}s".format(time.time() - trial_start)
        )

    LOGGER.info(
        "Batch Runner complete. Time elapsed: {:.1f}s".format(time.time() - start)
    )


if __name__ == "__main__":
    main()
