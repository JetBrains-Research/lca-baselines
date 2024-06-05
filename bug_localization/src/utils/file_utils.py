import os
from collections import Counter

from omegaconf import DictConfig, OmegaConf


def get_file_ext(filepath: str):
    return os.path.splitext(filepath)[-1].lower()


def get_file_exts(files: list[str]) -> dict[str, int]:
    return dict(Counter([get_file_ext(file) for file in files]))


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
    with open(os.path.join(path, 'config.yamls'), 'w') as f:
        f.write(OmegaConf.to_yaml(config))


def is_test_file(file_path: str):
    return any(test_dir in file_path.lower() for test_dir in ['test/', 'tests/'])
