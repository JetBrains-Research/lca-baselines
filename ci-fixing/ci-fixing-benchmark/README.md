## Config

To initialize the benchmark, you need to pass a path to a config file with the following fields (see example in **benchmark.yaml**):

**repos_folder**: here the cloned repos would be stored  
**out_folder**: here the result files would be stored  
**data_cache_dir**: here the cached dataset would be stored  
**username**: your GitHub username  
**test_username**: Username that would be displayed in the benchmark  
**language**: dataset language (now only Python is available)  

## Benchmark usage

To use the benchmark you need to pass a function that fixes the repo according 
the repo state on a local machine, logs and metadata of the failed workflows (**fix_repo_function**).

It should have the following (all optional) arguments:
(datapoint, repo_path, repo, out_folder)

**datapoint**:  dp from the dataset (its structure would be given below)  
**repo_path**:  path to the repo in the user's machine  
**repo**:       git.Repo object from GitPython library  
**out_folder**: folder for the benchmark results output  

For now, only two functions have been implemented:

**fix_none** -       does nothing  
**fix_apply_diff** - applies the diff that fixed the issue in the original repo  

## Evaluate dataset

method **CIFixBenchmark.eval_dataset(fix_repo_function)** evaluates dataset:

1. Downloads dataset (from https://huggingface.co/datasets/JetBrains-Research/lca-ci-fixing)
2. Sends the datapoints on GitHub to run workflows
3. Requests results from GitHub
4. Analyzes results and print them.

Further, we may duplicate the request part on our side.

## Outputs

Method's outputs:

1. **jobs_ids.jsonl** - identificators of the jobs that were sent to the GitHub. It is used for the further evaluation
2. **jobs_results.jsonl** - results of each job.
3. **jobs_awaiting.jsonl** - list of awaiting jobs (normally should be empty)
3. **jobs_invalid.jsonl** - list of invalid jobs (normally should be empty)

Examples of these files can be found in the /examples folder

You can also evaluate your results using method **CIFixBenchmark.eval_jobs(result_filename=result_filename)**
passing **jobs_ids.jsonl** file.

You can download the dataset using **CIFixBenchmark.get_dataset()** method (example at the end of the file)
