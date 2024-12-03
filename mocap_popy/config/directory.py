import os
from pathlib import Path


def find_root_dir():
    """Recursively search upwards for the project root directory."""
    marker = "mocap_popy"
    current_dir = Path(__file__).resolve().parent

    for parent in current_dir.parents:
        if (parent / marker).exists() and parent.name == marker:
            return parent

    raise RuntimeError(f"Root directory with marker '{marker}' not found.")


ROOT_DIR = find_root_dir()

# paths under root
SRC_DIR = os.path.join(ROOT_DIR, "mocap_popy")
CONFIG_DIR = os.path.join(ROOT_DIR, "config")
LOG_DIR = os.path.join(ROOT_DIR, "logs")
DATASET_DIR = os.path.join(ROOT_DIR, "example_datasets")
UTILS_DIR = os.path.join(SRC_DIR, "utils")
SCRIPTS_DIR = os.path.join(SRC_DIR, "scripts")
AUX_DIR = os.path.join(SRC_DIR, "aux_scripts")

# paths under SRC (excluding config)
TEMPLATES_DIR = os.path.join(SRC_DIR, "model_templates")
JSON_TEMPLATES_DIR = os.path.join(TEMPLATES_DIR, "json")
RIGID_BODY_JSON_DIR = os.path.join(JSON_TEMPLATES_DIR, "rigid_body")

TEMPLATE_FILE_MAPPING = os.path.join(TEMPLATES_DIR, "template_file_mapping.json")


def get_next_filename(
    dirpath: str, basename: str, file_ext: str, limit: int = 100
) -> str:
    """!Get the next available filename for a basename."""
    if os.path.exists(os.path.join(dirpath, f"{basename}.{file_ext}")):
        for i in range(1, limit):
            if not os.path.exists(os.path.join(dirpath, f"{basename}_{i}.{file_ext}")):
                basename = f"{basename}_{i}"
                break
    return os.path.join(dirpath, f"{basename}.{file_ext}")
