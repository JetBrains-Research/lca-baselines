import os
from omegaconf import OmegaConf

from benchmark import CIFixBenchmark
from benhmark_functions import fix_apply_diff, fix_none

"""
private
"""
#TODO add contact info
"""
Here the tokens are read.
First, ask us (placeholder for contact info) to add you to the benchmark-owner organization,
then you need your personal GH token to use the benchmark.
"""

token_gh = os.environ.get("TOKEN_GH")
config_private_path = "tokens_paths.yaml"
if os.path.exists(config_private_path):
    config_private = OmegaConf.load(config_private_path)

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

# pass your fixing function
# For debugging, please, limit yourself to a small amount of datapoints (argument num_dp)
fix_repo_function = fix_none  # fix_apply_diff  #
# CIBenchPython.eval_dataset(fix_repo_function, num_dp=5)

# Download the dataset if you want to play with it
test_dataset = CIBenchPython.get_dataset(force_download=False, num_dp=5)

# You can load datased from the local folder with json files, passing the path to an argument dataset_folder
# test_dataset = CIBenchPython.get_dataset(force_download=False, dataset_folder=dataset_folder)

# Evaluate jobs
# job_ids_file = "examples/jobs_ids.jsonl"
# job_results = CIBenchPython.eval_jobs(job_ids_file=job_ids_file, result_filename="jobs_results_test.jsonl")

# Analyze jobs
# job_results_file = "examples/jobs_results.jsonl"
# CIBenchPython.analyze_results(jobs_results_file=job_results_file)

pass
