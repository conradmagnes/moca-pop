import os

CONFIG_DIR = os.path.join(os.path.dirname(__file__))
SRC_DIR = os.path.dirname(CONFIG_DIR)
ROOT_DIR = os.path.dirname(SRC_DIR)

# paths under root
LOG_DIR = os.path.join(ROOT_DIR, "logs")
DATASET_DIR = os.path.join(ROOT_DIR, "example_datasets")

# paths under SRC (excluding config)
TEMPLATES_DIR = os.path.join(SRC_DIR, "templates")
RB_TEMPLATES_DIR = os.path.join(TEMPLATES_DIR, "rigid_body")
