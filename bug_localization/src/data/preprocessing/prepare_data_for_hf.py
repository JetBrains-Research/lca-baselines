import multiprocessing
import os

import hydra
import pandas as pd
from dotenv import load_dotenv
from omegaconf import DictConfig

from src.utils.hf_utils import CATEGORIES


def add_stats(dp, dp_info):
    dp['repo_symbols_count'] = dp_info['repo_symbols_count']
    dp['repo_tokens_count'] = dp_info['repo_tokens_count']
    dp['repo_lines_count'] = dp_info['repo_lines_count']
    dp['repo_files_without_tests_count'] = dp_info['repo_files_without_tests_count']

    dp['changed_symbols_count'] = dp_info['changed_symbols_count']
    dp['changed_tokens_count'] = dp_info['changed_tokens_count']
    dp['changed_lines_count'] = dp_info['changed_lines_count']
    dp['changed_files_without_tests_count'] = dp_info['changed_files_without_tests_count']

    dp['issue_symbols_count'] = dp_info['issue_symbols_count']
    dp['issue_words_count'] = dp_info['issue_words_count']
    dp['issue_tokens_count'] = dp_info['issue_tokens_count']
    dp['issue_lines_count'] = dp_info['issue_lines_count']
    dp['issue_links_count'] = dp_info['issue_links_count']
    dp['issue_code_blocks_count'] = dp_info['issue_code_blocks_count']

    return dp


def filter_outliers(df: pd.DataFrame) -> pd.DataFrame:
    df = df.dropna()
    # 0.99 quantile
    df = df[df['changed_files_count'] <= 22]
    # 0.99 quantile
    df = df[df['changed_lines_count'] <= 594]
    # Should not filter anything but for more sure
    df = df[df['changed_files_without_tests_count'] > 0]
    # 0.01 quantile
    df = df[df['issue_tokens_count'] >= 13]
    # 0.99 quantile
    df = df[df['issue_tokens_count'] <= 4491]

    df['repo_tokens_count'] = df['repo_tokens_count'].astype(int)
    df['changed_tokens_count'] = df['changed_tokens_count'].astype(int)
    df['issue_tokens_count'] = df['issue_tokens_count'].astype(int)
    return df


def prepare_dataset(config: DictConfig):
    stats_df = pd.read_csv(os.path.join(config.bug_localization_data_path, 'metrics.csv'))

    for category in CATEGORIES:
        df = pd.read_csv(os.path.join(config.bug_localization_data_path, f"bug_localization_data_{category}.csv"))

        params = [(dp, stats_df.loc[stats_df['text_id'] == dp["text_id"]].squeeze()) for _, dp in df.iterrows()]

        cpus = multiprocessing.cpu_count()
        with multiprocessing.Pool(processes=cpus) as pool:
            results = pool.starmap(add_stats, params)

        df = pd.DataFrame(results)
        df = filter_outliers(df)

        print(f"{category}: {len(df)}")
        df.to_csv(os.path.join(config.bug_localization_data_path, f"bug_localization_data_{category}.csv"),
                  escapechar="\\", index=False)
        df.to_json(os.path.join(config.bug_localization_data_path, f"bug_localization_data_{category}.jsonl"),
                   orient="records", lines=True)


@hydra.main(config_path="../../../configs/data", config_name="server", version_base=None)
def main(config: DictConfig):
    prepare_dataset(config)


if __name__ == "__main__":
    load_dotenv()
    main()
