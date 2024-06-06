from benchmark import CIFixBenchmark
from benhmark_functions import fix_apply_diff, fix_none

# Name of the model used for the benchmark. It is used to track evaluation
model_name = "diff"

# Creating benchmark object
config_path = "config.yaml"
CIBenchPython = CIFixBenchmark(model_name, config_path)

# pass your fixing function
# For debugging, please, limit yourself to a small amount of datapoints (argument num_dp)
# fix_repo_function = fix_none  # Avialable options: fix_apply_diff, fix_none
fix_repo_function = fix_apply_diff

# Run the benchmark
# CIBenchPython.eval_dataset(fix_repo_function, num_dp=None, ids_list=None)
# You can run this method after evaluating dataset if some datapoints remained in waiting list.
# CIBenchPython.get_results()

# Download the dataset if you want to play with it
# test_dataset = CIBenchPython.get_dataset(force_download=False)

# You can load datased from the local folder with json files, passing the path to an argument dataset_folder
# test_dataset = CIBenchPython.get_dataset(force_download=False, dataset_folder=dataset_folder)

# Evaluate jobs
# job_ids_file = "examples/jobs_ids.jsonl"
# job_results = CIBenchPython.eval_jobs(job_ids_file=job_ids_file, result_filename="jobs_results_test.jsonl")

# Analyze jobs
# job_results_file = "examples/jobs_results.jsonl"
# CIBenchPython.analyze_results(jobs_results_file=job_results_file)
