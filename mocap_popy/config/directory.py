import os

CONFIG_DIR = os.path.join(os.path.dirname(__file__))
SRC_DIR = os.path.dirname(CONFIG_DIR)
ROOT_DIR = os.path.dirname(SRC_DIR)
LOG_DIR = os.path.join(ROOT_DIR, "logs")
DATASET_DIR = os.path.join(ROOT_DIR, "example_datasets")
