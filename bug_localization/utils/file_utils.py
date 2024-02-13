import json
import os

from omegaconf import DictConfig, OmegaConf


def create_dir(dir_path: str) -> str:
    if not os.path.exists(dir_path):
        os.makedirs(dir_path, exist_ok=True)

    return dir_path


def create_run_directory(baseline_results_path: str) -> tuple[str, int]:
    run_index = 0
    while os.path.exists(os.path.join(baseline_results_path, f'run_{run_index}')):
        run_index += 1

    run_path = create_dir(os.path.join(baseline_results_path, f'run_{run_index}'))

    return run_path, run_index


def save_config(config: DictConfig, path: str):
    with open(os.path.join(path, 'config.yaml'), 'w') as f:
        f.write(OmegaConf.to_yaml(config))
