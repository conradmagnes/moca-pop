import json
import numpy as np


def convert_to_native_types(obj):
    if isinstance(obj, dict):
        return {k: convert_to_native_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_native_types(i) for i in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, float) and np.isnan(obj):
        return None
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj


def export_str_as_json(data, filename):
    with open(filename, "w") as f:
        json.dump(data, f, indent=4)


def export_dict_as_json(data, filename):
    with open(filename, "w") as f:
        json.dump(convert_to_native_types(data), f, indent=4)


def import_json_as_dict(filename):
    """!Import JSON file as dict"""
    json_dict = {}
    with open(filename, "r") as f:
        json_dict = json.load(f)
    return json_dict


def import_json_as_str(filename):
    """!Import JSON file as string"""
    json_str = ""
    with open(filename, "r") as f:
        json_str = f.read()

    return json_str
