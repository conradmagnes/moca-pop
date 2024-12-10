"""!
    Argparse utility functions
    ==========================

    Utility functions for parsing and validating argparse arguments.

    @author C. McCarthy
"""

import logging
import os

from moca_pop.config import directory
from moca_pop.utils import model_template_loader

LOGGER = logging.getLogger("ArgparseUtils")


def validate_offline_project_dir(project_name: str) -> str:
    """!Validate offline mode project directory.

    @param project_name Project directory or example dataset name.
    @return project_dir
    """

    if not project_name:
        LOGGER.error("Project directory must be provided in offline mode.")
        exit(-1)

    if "example_datasets" in project_name:
        project_dir = os.path.join(
            directory.DATASET_DIR, project_name.split(os.sep)[-1]
        )
    elif project_name in os.listdir(directory.DATASET_DIR):
        LOGGER.info("Found project directory in example datasets.")
        project_dir = os.path.join(directory.DATASET_DIR, project_name)
    else:
        project_dir = project_name

    if not os.path.isdir(project_dir):
        LOGGER.error(f"Project directory does not exist ({project_dir}). Exiting.")
        exit(-1)

    return project_dir


def validate_offline_paths(
    project_name: str, trial_name: str, subject_name: str
) -> tuple:
    """!Validate offline mode file paths.

    @param project_name Project directory or example dataset name.
    @param trial_name Trial name.
    @param subject_name Subject name.
    @return tuple of project_dir, trial_fp, vsk_fp, subject_name
    """
    project_dir = validate_offline_project_dir(project_name)

    if not trial_name or not subject_name:
        LOGGER.error(
            "Project directory, trial name, and subject name must be provided in offline mode."
        )
        exit(-1)

    trial_fp = os.path.join(project_dir, f"{trial_name}.c3d")
    if not os.path.isfile(trial_fp):
        LOGGER.error(f"Trial file does not exist ({trial_fp}). Exiting.")
        exit(-1)

    vsk_fp = os.path.join(project_dir, f"{subject_name}.vsk")
    if not os.path.isfile(vsk_fp):
        LOGGER.error(f"VSK file was not found: {vsk_fp}. Exiting.")
        exit(-1)

    return project_dir, trial_fp, vsk_fp, subject_name


def validate_online_subject_name(vicon, subject_name: str = None) -> str:
    """!Checks to make sure subject is preset and active in Nexus.

    If no subject name is provided, will search for a match in the available templates.

    @param vicon ViconNexus instance.
    @param subject_name Subject name.

    @return subject_name
    """

    subject_names, subject_templates, subject_statuses = vicon.GetSubjectInfo()

    if not subject_name:
        LOGGER.info("Searching for available subject templates...")
        candidate_subject_names = []
        mapper = model_template_loader.load_mapper()
        for name, template, status in zip(
            subject_names, subject_templates, subject_statuses
        ):
            if template in mapper:
                candidate_subject_names.append((name, status))
        if len(candidate_subject_names) == 0:
            LOGGER.error("No matching templates found for trial subjects. Exiting.")
            exit(-1)
        elif len(candidate_subject_names) > 1:
            candidate_subject_names = [
                sn for sn, status in candidate_subject_names if status
            ]
            if len(candidate_subject_names) > 1 or len(candidate_subject_names) == 0:
                LOGGER.error("Could not infer subject name (multiple or none active).")
                exit(-1)

        subject_name, _ = candidate_subject_names[0]

    elif subject_name not in subject_names:
        LOGGER.error(f"Subject name '{subject_name}' not found in Nexus. Exiting.")
        exit(-1)
    else:
        subject_status = subject_statuses[subject_names.index(subject_name)]
        if not subject_status:
            LOGGER.info(f"Subject '{subject_name}' is not active. Setting to active.")
            vicon.SetSubjectActive(subject_name, True)

    return subject_name


def validate_online_paths(vicon, subject_name: str = None) -> tuple:
    """!Validate online mode arguments.

    Checks loaded trial and subject names, and vsk file.

    @param vicon ViconNexus instance.
    @param subject_name Subject name. If not provided, will search for match in available templates.
    @return tuple of project_dir, trial_fp, vsk_fp, subject_name
    """
    project_dir, trial_name = vicon.GetTrialName()
    if not trial_name:
        LOGGER.error("Load a trial in Nexus before running 'online' mode.")
        exit(-1)

    trial_fp = os.path.join(project_dir, f"{trial_name}.c3d")

    subject_name = validate_online_subject_name(vicon, subject_name)

    vsk_fp = os.path.join(project_dir, f"{subject_name}.vsk")
    if not os.path.isfile(vsk_fp):
        LOGGER.error(f"VSK file was not found: {vsk_fp}. Exiting.")
        exit(-1)

    return project_dir, trial_fp, vsk_fp, subject_name


def validate_start_end_frames(
    start_frame: int, end_frame: int, frames: list[int]
) -> tuple[int, int]:
    """!Validate start and end frames.

    @param start_frame start frame
    @param end_frame end frame
    @param frames list of frames in the trial
    @return tuple of start_frame, end_frame
    """
    first_frame = frames[0]
    last_frame = frames[-1]

    if start_frame >= last_frame:
        LOGGER.error(
            f"Start frame {start_frame} is out of range ({first_frame}-{last_frame}). Exiting."
        )
        exit(-1)

    start = first_frame if start_frame < 0 else start_frame

    if end_frame < 0:
        end = last_frame
    elif end_frame < start:
        LOGGER.error(f"End frame {end_frame} is before start frame. Exiting.")
        exit(-1)
    elif end_frame >= last_frame:
        LOGGER.warning(
            f"End frame {end_frame} is out of range ({first_frame}-{last_frame}). Setting to end."
        )
        end = last_frame
    else:
        end = end_frame

    return start, end
