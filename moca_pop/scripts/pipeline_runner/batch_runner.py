"""!
    Batch Runner
    ============

    This script runs the pipeline runner on a batch of trials. It is intended to be run as a standalone script.
    The script extracts pipelines from a configuration file. By default, the script looks for a configuration file
    in the `config` directory of this script's parent.

    The export flag of the pipeline runner is set to true by default. Pipelines should include a pipeline step which
    exports the trial to C3D, or otherwise save the trial (not recommended).

    Usage:
    ------

    python batch_runner.py -cn <config_name> -pp <project_path> --exclude <trials_to_exclude>

    Example: 
        python batch_runner.py -v -l -cn "nushu_pipeline_series.json" -pp "D:\HPL\pipeline_test_2" --include "20241107T104520Z_step-forward-r,20241107T104613Z_step-forward-l"

    Options:
    --------
    Run 'python batch_runner.py -h' for options.

    Returns:
    --------
    0 if successful, -1 if failed.


    @author: C. McCarthy
"""

import argparse
import logging
import os
import sys
import time

from moca_pop.config import logger, directory
from moca_pop.utils import dist_utils
import moca_pop.scripts.pipeline_runner.pipeline_runner as npr

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
        "-cn",
        "--config_name",
        type=str,
        required=True,
        help="Name of the pipeline configuration file.",
    )
    parser.add_argument(
        "-pp",
        "--project_path",
        type=str,
        required=True,
        help="Path to the project directory",
    )
    parser.add_argument(
        "--include",
        type=str,
        default="",
        help="Comma-separated list of trials to include",
    )
    parser.add_argument(
        "--exclude",
        type=str,
        default="",
        help="Comma-separated list of trials to exclude",
    )

    parser.add_argument(
        "-sn",
        "--subject_name",
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

    config_path = npr.validate_config_name(args.config_name)

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
    included_trials = args.include.split(",") if args.include else []
    if included_trials:
        trials = [t for t in trials if t in included_trials]

    excluded_trials = args.exclude.split(",") if args.exclude else []
    if excluded_trials:
        trials = [t for t in trials if t not in excluded_trials]

    LOGGER.info(f"Trials: {', '.join(trials)}")

    pipeline_runner_path = os.path.join(
        directory.SCRIPTS_DIR, "pipeline_runner", "pipeline_runner.py"
    )

    common_args = ["--export", "-cn", config_path, "-pn", project_path]
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
            verbose=args.verbose,
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
    exit(0)


if __name__ == "__main__":
    main()
