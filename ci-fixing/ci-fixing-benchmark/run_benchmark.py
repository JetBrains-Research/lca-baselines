import os
from omegaconf import OmegaConf

from benhmark_functions import fix_none, fix_apply_diff
from benchmark import CIFixBenchmark
from benchmark_utils import get_token

"""
private
"""

"""
Here the tokens are read.
I registered new GitHub user and added it to the organization. For now only this user can use benchmark.
Therefore I placed the GH token to a shared folder on the server and username in the benchmark.yaml (timur-for-test)
Current token path - /mnt/data/shared-data/lca/CI-fix-benchmark/gh_timur-for-test.txt
Better option - ask me to add you to the organization and use your personal token.
Before that DO NOT pass your token. 
As our dataset is private, you should pass your HuggingFace token either from file or from ENV variable.
"""

# TODO move token ENV variable to config
config_private_path = "send_datapoint.yaml"
config_private = OmegaConf.load(config_private_path)
token_gh = os.environ.get("TOKEN_GH")
token_hf = os.environ.get("TOKEN_HF")

if token_hf is None:
    print("Reading HuggingFace token from file")
    token_hf = get_token(config_private.token_hf_path)
if token_gh is None:
    print("Reading GitHub token from file")
    with open(config_private.token_gh_path) as f:
        token_gh = f.read()

"""
Public part
"""

# Name of the model used for the benchmark. It is used to track evaluation
model_name = "none"

# Creating benchmark object
config_path = "benchmark.yaml"
CIBenchPython = CIFixBenchmark(model_name, config_path, token_gh)

fix_repo_function = fix_none  # fix_apply_diff  #
out_filename = f"jobs_results_{model_name}.jsonl"
# CIBenchPython.eval_dataset(fix_repo_function, hf_token=token_hf, result_filename=out_filename)

# Download dataset if you want to play with it
test_dataset = CIBenchPython.get_dataset(token_hf, force_download=False)

# You can load datased from local folder with json files, passing path to an argument dataset_folder
# dataset_folder = "/mnt/data/shared-data/lca/CI-fix-benchmark/datapoints_json_verified"
# test_dataset = CIBenchPython.get_dataset(token_hf, force_download=False, dataset_folder=dataset_folder)

# Evaluate jobs
# job_ids_file = "/mnt/data/shared-data/lca/CI-fix-benchmark/benchmark/out/jobs_ids.jsonl"
# job_results = CIBenchPython.eval_jobs(job_ids_file=job_ids_file, result_filename="jobs_results_test.jsonl")

# Analyze jobs
# job_results_file = "/mnt/data/shared-data/lca/CI-fix-benchmark/benchmark/out/jobs_results_none.jsonl"
# CIBenchPython.analyze_results(jobs_results_file=job_results_file)

pass
