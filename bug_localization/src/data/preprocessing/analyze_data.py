import multiprocessing
import os
from collections import defaultdict
from typing import Optional

import hydra
import pandas as pd
import tiktoken
from datasets import load_dataset
from dotenv import load_dotenv
from omegaconf import DictConfig

from src.data.preprocessing.utils import get_links, get_code_blocks
from src.utils.git_utils import get_repo_content_on_commit, parse_changed_files_and_lines_from_diff, \
    parse_changed_files_from_diff
from src.utils.hf_utils import CATEGORIES

tokenizer = tiktoken.encoding_for_model('gpt-4')


def _get_changed_lines_content(changed_lines: list[tuple[int, str, str]]):
    return [changed_line[2] for changed_line in changed_lines]


def count_changed_symbols(changed_files_and_lines: dict[str, list[tuple[int, str, str]]]):
    return sum(count_symbols('\n'.join(_get_changed_lines_content(changed_lines)))
               for changed_lines in changed_files_and_lines.values())


def count_changed_tokens(changed_files_and_lines: dict[str, list[tuple[int, str, str]]]):
    try:
        return sum(count_tokens('\n'.join(_get_changed_lines_content(changed_lines)))
                   for changed_lines in changed_files_and_lines.values())
    except Exception as e:
        print(e)


def count_changed_lines(changed_files_and_lines: dict[str, list[tuple, tuple]]):
    return sum(len(_get_changed_lines_content(changed_lines))
               for changed_lines in changed_files_and_lines.values())


def count_repo_symbols(content: dict[str, str]):
    return sum([count_symbols(content) for content in content.values() if content])


def count_repo_tokens(content: dict[str, str]) -> Optional[int]:
    try:
        return sum([count_tokens(content) for content in content.values() if content])
    except Exception as e:
        print(e)
    return None


def count_repo_lines(content: dict[str, str]):
    return sum(count_lines(content) for content in content.values() if content)


def count_symbols(text: str) -> int:
    return len(text)


def count_tokens(text: str) -> Optional[int]:
    try:
        return len(tokenizer.encode(text))
    except Exception as e:
        print(e)
    return None


def count_lines(text: str) -> int:
    return len(text.split('\n'))


def count_words(text: str) -> int:
    return len(text.split())


def add_stats(config: DictConfig, dp, category: str):
    print(f"Processing {dp['text_id']}")
    repo_path = os.path.join(config.repos_path, f"{dp['repo_owner']}__{dp['repo_name']}")
    extensions = None if category == 'mixed' else [category]
    repo_content = get_repo_content_on_commit(repo_path, dp["base_sha"], extensions=extensions, ignore_tests=True)

    changed_files_and_lines = parse_changed_files_and_lines_from_diff(dp['diff'])
    changed_files_and_lines = {f: d for f, d, in changed_files_and_lines.items() if f in repo_content}
    changed_files = parse_changed_files_from_diff(dp['diff'])
    changed_files = [f for f in changed_files if f in repo_content]

    dp['repo_symbols_count'] = count_repo_symbols(repo_content)
    dp['repo_tokens_count'] = count_repo_tokens(repo_content)
    dp['repo_lines_count'] = count_repo_lines(repo_content)
    dp['repo_files_without_tests_count'] = len(repo_content)

    dp['changed_symbols_count'] = count_changed_symbols(changed_files_and_lines)
    dp['changed_tokens_count'] = count_changed_tokens(changed_files_and_lines)
    dp['changed_lines_count'] = count_changed_lines(changed_files_and_lines)
    dp['changed_files_without_tests_count'] = len(changed_files)

    issue_text = dp['issue_body']
    dp['issue_symbols_count'] = count_symbols(issue_text)
    dp['issue_tokens_count'] = count_tokens(issue_text)
    dp['issue_lines_count'] = count_lines(issue_text)
    dp['issue_words_count'] = count_words(issue_text)
    dp['issue_links_count'] = len(get_links(dp['issue_body']))
    dp['issue_code_blocks_count'] = len(get_code_blocks(dp['issue_body']))

    return dp


def add_stats_to_repo_data(config, dps: list[tuple[dict, str]]):
    return [add_stats(config, dp, category) for dp, category in dps]


def calc_stats(config: DictConfig):
    pds_by_repo = defaultdict(list)
    for category in CATEGORIES:
        df = load_dataset('json', split='train', data_files=os.path.join(config.bug_localization_data_path,
                                                                         f"bug_localization_data_{category}.jsonl"))
        for dp in df:
            pds_by_repo[f"{dp['repo_owner']}__{dp['repo_name']}"].append((dp, category))

    cpus = multiprocessing.cpu_count()
    params = [(config, dps) for dps in pds_by_repo.values()]

    with multiprocessing.Pool(processes=cpus) as pool:
        results = pool.starmap(add_stats_to_repo_data, params)

    results = [dp for dps in results for dp in dps]

    df = pd.DataFrame(results)
    df.to_csv(os.path.join(config.bug_localization_data_path, 'metrics.csv'), escapechar="\\", index=False)


@hydra.main(config_path="../../../configs/data", config_name="server", version_base=None)
def main(config: DictConfig):
    calc_stats(config)


if __name__ == "__main__":
    load_dotenv()
    main()
