"""!
    Template Loader
    ===============

    Functions to load templates from the template directory.

    @author C. McCarthy
"""

import logging
import os

LOGGER = logging.getLogger(__name__)


import mocap_popy.config.directory as directory
from mocap_popy.aux_scripts.generate_template_file_mapping import (
    generate_template_json_mapping,
)
import mocap_popy.utils.json_utils as json_utils

import mocap_popy.model_templates.rigidBodyTemplate as rbt


def load_mapper():
    """Load the template mapper from the template directory."""
    mapper_fp = directory.TEMPLATE_FILE_MAPPING
    if not os.path.exists(mapper_fp):
        LOGGER.info(f"Template directory mapper not found. Generating...")
        generate_template_json_mapping()

    try:
        mapper = json_utils.import_json_as_dict(mapper_fp)
    except ValueError as e:
        LOGGER.error(f"Error loading template mapper: {e}")
        return None

    return mapper


def load_template_from_json(model_name: str, template_type: str, template_name: str):
    """Return a template object using a template stored as a json file."""
    if template_type != "rigid_body":
        LOGGER.warning(f"Template type {template_type} not supported.")
        return None

    mapper = load_mapper()
    if mapper is None:
        LOGGER.error("No template mapper found. Exiting.")
        return None

    if model_name not in mapper:
        LOGGER.error(f"Model {model_name} not found in template mapper.")
        return None

    if template_type not in mapper[model_name]:
        LOGGER.error(f"Template type {template_type} not found for model {model_name}.")
        return None

    if template_name not in mapper[model_name][template_type]:
        LOGGER.error(
            f"Template {template_name} not found for model {model_name}, type {template_type}."
        )
        return None

    rel_template_fp = mapper[model_name][template_type][template_name]
    template_fp = os.path.join(
        directory.JSON_TEMPLATES_DIR, template_type, rel_template_fp
    )
    template_data = json_utils.import_json_as_str(template_fp)
    try:
        return rbt.RigidBodyTemplate.model_validate_json(template_data)
    except ValueError as e:
        LOGGER.error(f"Error loading template from {template_fp}: {e}")
        return None
