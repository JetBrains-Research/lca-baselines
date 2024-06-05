import os
import json
from pathlib import Path

from benchmark_utils import read_jsonl


def recalc_results(dataset_folder, results_folder):

    json_files = list(dataset_folder.glob('*.json'))
    dp_shas = []
    for json_file in json_files:
        with open(json_file, 'r') as f:
            data = json.load(f)
        dp_shas.append(data["sha_fail"])

    dp_shas = set(dp_shas)

    results_files = list(results_folder.glob('*.jsonl'))

    for results_file in results_files:
        results = read_jsonl(results_file)
        model_name = results_file.stem.split("_")[-1]
        conclusions = [result["conclusion"] for result in results if result["sha_original"] in dp_shas]
        succ_rate = sum([conclusion == "success" for conclusion in conclusions])/len(conclusions)
        print(f"Model = {model_name}, num items = {len(conclusions)}, success rate = {succ_rate:.3f}")


if __name__ == "__main__":
    dataset_folder = Path('/mnt/data/galimzyanov/data/LCA/HF_dataset/lca-ci-fixing_filtered')
    results_folder = Path('/mnt/data/galimzyanov/data/LCA/benchmark/out/jobs_results_models')
    recalc_results(dataset_folder, results_folder)
