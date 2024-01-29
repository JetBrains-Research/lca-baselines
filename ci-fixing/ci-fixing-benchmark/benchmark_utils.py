import json
import os
import shutil

from omegaconf import OmegaConf


def read_jsonl(file_path):
    data = []
    with open(file_path, "r") as f:
        for line in f:
            data.append(json.loads(line))
    return data


def save_jsonl(file_path, data):
    with open(file_path, "w") as f:
        for entry in data:
            json.dump(entry, f)
            f.write("\n")


def get_token_gh(config_path):
    config_private = OmegaConf.load(config_path)
    with open(config_private.token_gh_path) as f:
        token_gh = f.read()
    return token_gh


def get_token_hf(config_path):
    config_private = OmegaConf.load(config_path)
    token_hf = get_token(config_private.token_hf_path)
    return token_hf


def get_token(token_path):
    with open(token_path) as f:
        token = f.read()

    return token


def filter_out_res(data_folder, out_folder):
    """
    filter acording of results benchmarks
    """
    results_none_path = os.path.join(out_folder, "jobs_results_none.jsonl")
    results_diff_path = os.path.join(out_folder, "jobs_results_diff.jsonl")
    results_none = read_jsonl(results_none_path)
    results_diff = read_jsonl(results_diff_path)
    orig_path = os.path.join(data_folder, "datapoints_json_verified")
    filtered_path = os.path.join(data_folder, "datapoints_json_filtered")
    os.makedirs(filtered_path, exist_ok=True)
    original_sha = {
        result["sha_original"][:7]
        for result in results_none
        if result["conclusion"] == "failure"
    }
    fixed_sha = {
        result["sha_original"][:7]
        for result in results_diff
        if result["conclusion"] == "success"
    }

    sha_valid = original_sha.intersection(fixed_sha)

    for sha in sha_valid:
        dp_file = os.path.join(orig_path, f"{sha}.json")
        dp_filtered = os.path.join(filtered_path, f"{sha}.json")
        shutil.copy2(dp_file, dp_filtered)
