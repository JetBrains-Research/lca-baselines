import multiprocessing
from typing import Callable, Any, Optional

from omegaconf import DictConfig

from src.utils.jsonl_utils import get_repos


def process_repos_data(
        process_repo: Callable[[dict, DictConfig], Any],
        config: DictConfig,
        processes: Optional[int] = None,
) -> None:
    """
    Runs `process_repo` on each repo from file located in `repos_path`. Use for workers bounding task.
    :param process_repo: func that takes owner, name and optional token as input and does some processing task on repo
    :param config: config with run parameters
    :param processes: number of processes to run analysis
    """
    params = [(repo, config) for repo in get_repos(config.repos_list_path)]
    cpus = multiprocessing.cpu_count() if processes is None else processes
    assert cpus > 0

    with multiprocessing.Pool(processes=cpus) as pool:
        pool.starmap(process_repo, params)
