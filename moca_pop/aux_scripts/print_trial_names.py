"""!
    Print Trial Names
    =================

    This script prints the names of the trials in the project directory.
    Trials are identified by the presence of a '.Trial.enf' file in the project directory.
    Trials can also be filtered by whether they have been processed or not (i.e. whether a '.C3D' file exists).
    Alternatively, the script can write the trial names to a text file (rather than printing to the console).

    This script can be run in 'online' or 'offline' mode. In 'offline' mode, a project directory must be provided.
    In 'online' mode, a trial must be loaded in Vicon Nexus.

    Usage:
    ------
    python print_trial_names.py -pn /path/to/project/directory -wo --only_processed

    Options:
    --------
    -off, --offline: Run the script in offline mode. 
    -pn, --project_name: Path to the project directory. 
    -wo, --write_out: Write the trial names to a text file. 
    --only_processed: Only print processed trials. 
    --only_unprocessed: Only print unprocessed trials.
    --oneline: Print / write the trial names on one line.

    Returns:
    --------
    0 if successful, -1 if failed.

    @author C. McCarthy    
"""

import argparse
import logging
import os

from moca_pop.config import logger
from moca_pop.utils import argparse_utils


def write_file(file_path, trial_names, oneline):
    """!Write the trial names to a text file.

    @param file_path Path to the output file.
    @param trial_names List of trial names.
    @param oneline Write the trial names on one line (comma separated).
    """
    with open(file_path, "w") as f:
        if oneline:
            f.write(",".join(trial_names))
        else:
            for trial in trial_names:
                f.write(f"{trial}\n")


def configure_parser():
    """!Configure the argument parser."""
    parser = argparse.ArgumentParser(
        description="Print the names of the trials in the project directory."
    )

    parser.add_argument(
        "-off",
        "--offline",
        action="store_true",
        help="Run the script in offline mode.",
    )

    parser.add_argument(
        "-pn",
        "--project_name",
        type=str,
        help="Path to the project directory",
    )

    parser.add_argument(
        "-wo",
        "--write_out",
        action="store_true",
        help="Write the trial names to a text file.",
    )

    parser.add_argument(
        "--only_processed",
        action="store_true",
        help="Only print processed trials.",
    )

    parser.add_argument(
        "--only_unprocessed",
        action="store_true",
        help="Only print unprocessed trials.",
    )
    parser.add_argument(
        "--oneline",
        action="store_true",
        help="Print / write the trial names on one line.",
    )

    return parser


def main():

    parser = configure_parser()
    args = parser.parse_args()

    logger.set_root_logger(name="print-trial-names", mode="off")
    logging.root.info("Starting PrintTrialNames.")

    if args.only_processed and args.only_unprocessed:
        logging.root.error(
            "Cannot specify both --only_processed and --only_unprocessed."
        )
        exit(-1)

    if args.offline or args.project_name:
        project_dir = argparse_utils.validate_offline_project_dir(args.project_name)
    else:
        from viconnexusapi import ViconNexus

        vicon = ViconNexus()
        project_dir, _ = vicon.GetTrialName()

    project_files = os.listdir(project_dir)
    trial_names = [x.split(".")[0] for x in project_files if x.endswith(".Trial.enf")]

    if args.only_processed:
        trial_names = [
            x
            for x in trial_names
            if os.path.isfile(os.path.join(project_dir, f"{x}.c3d"))
        ]

    if args.only_unprocessed:
        trial_names = [
            x
            for x in trial_names
            if not os.path.isfile(os.path.join(project_dir, f"{x}.c3d"))
        ]

    if args.write_out:
        output_fp = os.path.join(project_dir, "trial_names.txt")
        write_file(output_fp, trial_names, args.oneline)
        logging.root.info(f"Trial names written to: {output_fp}")
    elif args.oneline:
        print(",".join(trial_names))
    else:
        for trial in trial_names:
            print(trial)

    logging.root.info("Done.")
    exit(0)


if __name__ == "__main__":
    main()
