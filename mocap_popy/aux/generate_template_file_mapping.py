"""!   
    Template File Mapping Generator
    ===============================

    Iterates through the template directory and generates a json file that maps model and template names 
    to template file paths.

    Usage:
    ------
    python generate_template_file_mapping.py [-h] [-v] [--log] [--file_type FILE_TYPE]

    Options:
    --------
    -h, --help
        Show this help message and exit.
    -v, --verbose
        Increase logging verbosity to debug.
    --log
        Log output to file.
    --file_type FILE_TYPE
        File type for template files. Currently only json is supported.

    Returns:
    --------
    0 if successful, -1 if no templates are found.

"""

import argparse
import logging
import os

import mocap_popy.config.directory as directory
import mocap_popy.config.logger as logger
import mocap_popy.utils.json_utils as json_utils

LOGGER = logging.getLogger("GenerateTemplateFileMapping")


def generate_template_json_mapping():
    """Generate a json file that maps model and body names to their respective templates."""

    LOGGER.info("Generating template json mapping...")

    mapping = {}

    template_subdir_names = os.listdir(directory.JSON_TEMPLATES_DIR)

    for subdir_name in template_subdir_names:
        subdir = os.path.join(directory.JSON_TEMPLATES_DIR, subdir_name)
        if not os.path.isdir(subdir):
            continue
        template_files = [
            os.path.join(subdir, file)
            for file in os.listdir(subdir)
            if file.endswith(".json")
        ]
        for tp in template_files:
            template = json_utils.import_json_as_dict(tp)
            template_name = template.get("name", None)
            if template_name is None:
                LOGGER.warning(f"Template at {tp} does not have a name.")
                continue

            parent_models = template.get("parent_models", [])
            if not parent_models:
                LOGGER.info(
                    f"Template {template_name} does not have parent models. Adding to 'undefined_model_templates'."
                )
                parent_models = ["undefined_model_templates"]

            for model in parent_models:
                if model not in mapping:
                    mapping[model] = {}

                if subdir_name not in mapping[model]:
                    mapping[model][subdir_name] = {}

                if template_name in mapping[model][subdir_name]:
                    LOGGER.warning(
                        f"Template {template_name} already exists for model {model}. Overwriting."
                    )
                mapping[model][subdir_name].update({template_name: tp})

    if not mapping:
        LOGGER.error("No templates found.")
        return -1

    json_utils.export_dict_as_json(mapping, directory.TEMPLATE_FILE_MAPPING)
    LOGGER.info(f"Template file mapping saved to {directory.TEMPLATE_FILE_MAPPING}")
    return 0


def configure_parser():
    parser = argparse.ArgumentParser(
        description="Iterates through the template directory and generates a json file that maps model and template names to template file paths."
    )

    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Increase logging verbosity to debug.",
    )

    parser.add_argument(
        "--log",
        action="store_true",
        help="Log output to file.",
    )

    parser.add_argument(
        "--file_type",
        type=str,
        default="json",
        help="File type for template files. Currently only json is supported.",
    )

    return parser


if __name__ == "__main__":

    parser = configure_parser()
    args = parser.parse_args()

    mode = "w" if args.log else "off"
    logger.set_root_logger(name="generate_template_file_mapping", mode=mode)

    if args.verbose:
        logger.set_global_logging_level(logging.DEBUG)

    LOGGER.debug(f"Arguments: {args}")

    if args.file_type != "json":
        LOGGER.error(f"File type {args.file_type} is not supported.")
        exit()

    res = generate_template_json_mapping()
