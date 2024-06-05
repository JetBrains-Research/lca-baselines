import os
import json
from tqdm import tqdm
from pathlib import Path

from benchmark_utils import read_jsonl


def filter_dps(dataset_folder, results_file):

    json_files = list(dataset_folder.glob('*.json'))
    results = read_jsonl(results_file)
    fail_ids = [result["id"] for result in results if result["conclusion"] == "failure"]
    dp_num = 0

    for json_file in json_files:
        with open(json_file, 'r') as f:
            data = json.load(f)
        id = data["id"]

        if id in fail_ids:
            dp_num += 1
            os.remove(json_file)

    print(f"{dp_num} files deleted")

def reindex_dps(dataset_folder, results_file):

    json_files = list(dataset_folder.glob('*.json'))
    results = read_jsonl(results_file)
    fail_ids = [result["id"] for result in results if result["conclusion"] == "failure"]
    dp_num = 0

    for json_file in json_files:
        with open(json_file, 'r') as f:
            data = json.load(f)
        id = data["id"]

        if id in fail_ids:
            dp_num += 1
            os.remove(json_file)

    print(f"{dp_num} files deleted")


if __name__ == "__main__":
    dataset_folder = Path('/mnt/data/galimzyanov/data/LCA/HF_dataset/lca-ci-fixing_filtered')
    results_file = Path('/mnt/data/galimzyanov/data/LCA/benchmark/out/jobs_results_diff_28_05_1300.jsonl')
    # filter_dps(dataset_folder, results_file)
