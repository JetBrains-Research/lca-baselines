import os
import shutil
import tarfile
from typing import List, Optional

import huggingface_hub
from datasets import get_dataset_config_names, load_dataset
from huggingface_hub import hf_hub_download

from .base_data_source import BaseDataSource
from src.utils.git_utils import get_repo_content_on_commit, get_changed_files_between_commits
from src.utils.hf_utils import HUGGINGFACE_REPO, FEATURES, CATEGORIES


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
            data_files=f"repos_paths.json",
            ignore_verifications=True,
            split="train",
            features=FEATURES['repos_paths']
        )

        local_repo_tars_path = os.path.join(self._repos_dir, "local_repos_tars")

        # Load each repo in .tar.gz format, unzip, delete archive
        for category in CATEGORIES:
            repos = paths_json[category][0]
            for i, repo_tar_path in enumerate(repos):
                print(f"Loading {i}/{len(repos)} {repo_tar_path}")

                repo_name = os.path.basename(repo_tar_path)
                if os.path.exists(os.path.join(self._repos_dir, repo_name)):
                    print(f"Repo {repo_tar_path} is already loaded...")
                    continue

                local_repo_tar_path = hf_hub_download(
                    HUGGINGFACE_REPO,
                    filename=repo_tar_path,
                    repo_type='dataset',
                    local_dir=local_repo_tars_path,
                )

                with tarfile.open(local_repo_tar_path, "w:gz") as tar:
                    for file_ in tar:
                        try:
                            tar.extract(file_)
                        except Exception as e:
                            print(e)
                            os.remove(file_.name)
                            tar.extract(file_)
                        finally:
                            os.chmod(file_.name, 0o777)

                shutil.unpack_archive(local_repo_tar_path, extract_dir=self._repos_dir, format='gztar')
                os.remove(local_repo_tar_path)
        shutil.rmtree(local_repo_tars_path)

    def __iter__(self):
        for config in self._configs:
            dataset = load_dataset(self._hub_name, config, split=self._split, cache_dir=self._cache_dir)
            # TODO: fix loading of repos and tar.gz opening
            # self._load_repos()
            for dp in dataset:
                repo_path = os.path.join(self._repos_dir, f"{dp['repo_owner']}__{dp['repo_name']}")
                extensions = [config] if config != 'mixed' else None
                # Move parameters to data source config
                repo_content = get_repo_content_on_commit(repo_path, dp['base_sha'],
                                                          extensions=extensions,
                                                          ignore_tests=True)
                changed_files = get_changed_files_between_commits(repo_path, dp['base_sha'], dp['head_sha'],
                                                                  extensions=extensions,
                                                                  ignore_tests=True)
                yield dp, repo_content, changed_files
