# 🏟️ Long Code Arena Baselines
## CI builds repair

This directory contains the code for the CI builds repair benchmark. 

## ⚙️ Config

To initialize the benchmark, you need to pass a path to a config file with the following fields (see example in [`config_template.yaml`](config_template.yaml)):

`repos_folder`: the path to where the cloned repos will be stored;
`out_folder`: the path to where the result files will be stored;  
`data_cache_dir`: the path to where the cached dataset will be stored;  
`username_gh`: your GitHub username;  
`test_username`: _Optional_. Username that would be displayed in the benchmark, if ommitted, `username_gh` will be used;  
`language`: dataset language (for now, only Python is available).  

## 🏟️ Benchmark usage

For the example of the benchmark usage code, see the [`run_benchmark.py`](run_benchmark.py) script.
To use the benchmark, you need to pass a function `fix_repo_function` that fixes the build according to 
the repository state on a local machine, logs, and the metadata of the failed workflows.
The function should have the following (all optional) arguments:
(`datapoin`, `repo_path`, `repo`, `out_folder`)

`datapoint`:  datapoint from the dataset (its structure is given below);  
`repo_path`:  path to the repo on the user's machine;  
`repo`:       git.Repo object from GitPython library;  
`out_folder`: directory for outputting the benchmark results.  

For now, only two functions have been implemented:

`fix_none` —       does nothing;  
`fix_apply_diff` — applies the diff that fixed the issue in the original repository;  

## 🚀 Evaluate the baseline

The method `CIFixBenchmark.eval_dataset(fix_repo_function)` evaluates the baseline function. Specifically, it:

1. Downloads the [dataset](https://huggingface.co/datasets/JetBrains-Research/lca-ci-builds-repair);
2. Sends the datapoints to GitHub to run workflows;
3. Requests results from GitHub;
4. Analyzes results and prints them.

For debugging, please limit yourself to a small number of datapoints (argument `num_dp=num_dp`).

## 📈 Outputs

The evaluation method outputs the following results:

1. `jobs_ids.jsonl` — identifiers of the jobs that were sent to GitHub. They are used for the further evaluation.
2. `jobs_results.jsonl` — results of each job.
3. `jobs_awaiting.jsonl` — list of awaiting jobs (normally should be empty).
3. `jobs_invalid.jsonl` — list of invalid jobs (normally should be empty).

Examples of these files can be found in the (`/examples`)[examples/] directory.

You can also evaluate your results using the method `CIFixBenchmark.eval_jobs(result_filename=result_filename)`,
passing the `jobs_ids.jsonl` file. You can download the dataset using the `CIFixBenchmark.get_dataset()` method.
