import json
import os
from types import SimpleNamespace


def load_model_config(config_model_path):

    # Check if the config file exists
    if not os.path.exists(config_model_path):
        raise FileNotFoundError(f"Model config file not found: {config_model_path}")

    # Load the config file
    with open(config_model_path, 'r') as f:
        config_dict = json.load(f)

    return config_dict


def load_experiment_config(config_experiment_path):
    # Check if the config file exists
    if not os.path.exists(config_experiment_path):
        raise FileNotFoundError(f"Experiment config file not found: {config_experiment_path}")

    # Load the config file
    with open(config_experiment_path, 'r') as f:
        config_dict = json.load(f)

    return config_dict


def merge_configs(config_model_path, config_experiment_path):
    flat_config = {}

    # Add model config (top level)
    for key, value in config_model_path.items():
        if not isinstance(value, dict):
            flat_config[key] = value

    # Add experiment config (top level)
    for key, value in config_experiment_path.items():
        if not isinstance(value, dict) and key not in flat_config:
            flat_config[key] = value

    # Add nested attributes from model config
    # (None expected in current implementation)

    # Add nested attributes from experiment config
    for section_name, section_dict in config_experiment_path.items():
        if isinstance(section_dict, dict):
            for key, value in section_dict.items():
                flat_config[key] = value

    # Convert to SimpleNamespace (similar to argparse.Namespace)
    args = SimpleNamespace(**flat_config)

    # Add device_ids for compatibility
    if hasattr(args, 'use_multi_gpu') and args.use_multi_gpu and hasattr(args, 'devices'):
        device_ids = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in device_ids]
    else:
        args.device_ids = [0]

    # Set model name as model attribute for compatibility
    if hasattr(args, 'model_name') and not hasattr(args, 'model'):
        args.model = args.model_name

    return args


def load_config(config_name='default_config', experiment_name='default'):
    """
    Load and merge model and experiment configurations.

    Args:
        config_name (str, optional): Name of the model config file without extension. Defaults to 'default_config'.
        experiment_name (str, optional): Name of the experiment config file without extension. Defaults to 'default'.

    Returns:
        SimpleNamespace: An object with attributes corresponding to the merged config parameters
    """
    model_config = load_model_config(config_name)
    experiment_config = load_experiment_config(experiment_name)

    return merge_configs(model_config, experiment_config)
