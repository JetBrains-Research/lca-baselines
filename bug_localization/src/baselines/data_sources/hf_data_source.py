import os
import zipfile
from typing import List, Optional

import huggingface_hub
from datasets import get_dataset_config_names, load_dataset
from huggingface_hub import hf_hub_download

from src.baselines.data_sources.base_data_source import BaseDataSource
from src.utils.git_utils import get_repo_content_on_commit, get_changed_files_between_commits
from src.utils.hf_utils import HUGGINGFACE_REPO, FEATURES


class HFDataSource(BaseDataSource):

    def __init__(
            self,
            hub_name: str,
            repos_dir: str,
            configs: Optional[List[str]] = None,
            split: Optional[str] = None,
            cache_dir: Optional[str] = None,
    ):
        self._hub_name = hub_name
        self._cache_dir = cache_dir
        self._repos_dir = repos_dir

        if configs:
            self._configs = configs
        else:
            self._configs = get_dataset_config_names(self._hub_name)
        self._split = split

    def _load_repos(self):
        huggingface_hub.login(token=os.environ['HUGGINGFACE_TOKEN'])

        # Load json file with repos paths
        paths_json = load_dataset(
            HUGGINGFACE_REPO,
            data_files=f"repos.json",
            ignore_verifications=True,
            split="train",
            features=FEATURES['repos']
        )

        local_repo_zips_path = os.path.join(self._repos_dir, "local_repos_zips")

        # Load each repo in .zip format, unzip, delete archive
        for category in self._configs:
            repos = paths_json[category][0]
            for i, repo_zip_path in enumerate(repos):
                print(f"Loading {i}/{len(repos)} {repo_zip_path}")

                repo_name = os.path.basename(repo_zip_path).split('.zip')[0]
                repo_path = os.path.join(self._repos_dir, repo_name)
                if os.path.exists(os.path.join(self._repos_dir, repo_name)):
                    print(f"Repo {repo_zip_path} is already loaded...")
                    continue

                local_repo_zip_path = hf_hub_download(
                    HUGGINGFACE_REPO,
                    filename=repo_zip_path,
                    repo_type='dataset',
                    local_dir=local_repo_zips_path,
                )

                with zipfile.ZipFile(local_repo_zip_path, 'r') as zip_ref:
                    zip_ref.extractall(repo_path)
                os.remove(local_repo_zip_path)

    def __iter__(self):
        for config in self._configs:
            dataset = load_dataset(self._hub_name, config, split=self._split, cache_dir=self._cache_dir)
            self._load_repos()
            for dp in dataset:
                repo_path = os.path.join(self._repos_dir, f"{dp['repo_owner']}__{dp['repo_name']}")
                try:
                    repo_content = get_repo_content_on_commit(repo_path, dp['base_sha'],
                                                              extensions=[config],
                                                              ignore_tests=True)

                    changed_files = get_changed_files_between_commits(repo_path, dp['base_sha'], dp['head_sha'],
                                                                      extensions=[config],
                                                                      ignore_tests=True)
                    yield dp, repo_content, changed_files
                except Exception as e:
                    print(f"Failed to get repo content for {dp['repo_owner']}__{dp['repo_name']}", e)
                    continue
