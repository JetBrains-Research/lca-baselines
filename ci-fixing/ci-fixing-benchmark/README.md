To initialize benchmark, you need to pass a path to a config file with following fields:

repos_folder: here the cloned repos would be stored 
out_folder: here the result files would be stored
data_cache_dir: here the cached dataset would be stored
username: your GitHub username
test_username: Username that would be displayed in the benchmark
language: # dataset language (now only Python is available)

To use the benchmark you need to pass a function that fixes the repo according 
the repo state and logs and metadata of the failed workflows (fix_repo_function).

It should have following (all optional) arguments:
(datapoint, repo_path, repo, out_folder)

datapoint:  dp from the dataset (its structure would be given below)
repo_path:  path to the repo in the user's machine
repo:       git.Repo object from GitPython library
out_folder: folder for the benchmark results output

For now, I implemented only two functions:

fix_none -       does nothing
fix_apply_diff - applies the diff that fixed issue in the original repo


method CIFixBenchmark.eval_dataset(fix_repo_function) evaluates dataset:

1. Downloads dataset (from https://huggingface.co/datasets/JetBrains-Research/lca-ci-fixing)
2. Sends the datapoints on GitHub to run workflows
3. Requests results from GitHub
4. Analyzes results and prints them.

Further, maybe we will duplicate request part at our side.

Method's outputs:

1. jobs_ids.jsonl - identificators of the jobs that were sent to the GitHub. It is used for the further evaluation
2. jobs_results.jsonl - results of each job.
3. jobs_awaiting.jsonl - list of awaiting jobs (normally should be empty)
3. jobs_invalid.jsonl - list of invalid jobs (normally should be empty)

Examples can be found in examples folder /mnt/data/shared-data/lca/CI-fix-benchmark/benchmark/out

You can also just evaluate your results using method CIFixBenchmark.eval_jobs(result_filename=result_filename)
passing jobs_ids.jsonl file.

You can download dataset using CIFixBenchmark.get_dataset() method (example in the end of the file)
