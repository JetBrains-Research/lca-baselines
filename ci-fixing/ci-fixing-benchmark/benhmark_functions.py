import os

import git
import requests
from git import GitCommandError
from ruamel.yaml import YAML


def edit_workflow_push(workflow_file):
    """
    editing workflow.yaml so, that it would be run on push
    """

    yaml = YAML()
    with open(workflow_file, "r") as file:
        yaml_data = yaml.load(file)

    yaml_data["on"] = "push"

    with open(workflow_file, "w") as file:
        yaml.dump(yaml_data, file)


def copy_and_edit_workflow_file(datapoint, repo):
    """
    Copy workflow.yaml from gathered data to the repo
    Delete all other workflows.
    """
    workflow_dir = os.path.join(repo.working_dir, ".github/workflows")
    for filename in os.listdir(workflow_dir):
        file_path = os.path.join(workflow_dir, filename)
        if os.path.isfile(file_path):
            os.remove(file_path)
    workflow_file = os.path.join(workflow_dir, "workflow.yaml")
    with open(workflow_file, "w") as f:
        f.write(datapoint["workflow"])
    edit_workflow_push(workflow_file)


def rename_precommit_files(repo_path):
    """
    rename pre-commit.yaml, so it will be run on push
    """
    workflow_dir = os.path.join(repo_path, ".github/workflows")
    for filename in os.listdir(workflow_dir):
        file_path = os.path.join(workflow_dir, filename)
        if os.path.isfile(file_path):
            if "pre-commit" in filename.lower():
                os.rename(
                    file_path, file_path.lower().replace("pre-commit", "precommit")
                )


def push_repo(repo, credentials, benchmark_owner, user_branch_name):
    """
    Pushes the corrected repo, return commit sha to use it for getting results
    """

    # TODO think about adding only changed files
    repo.git.add(".")
    repo.git.add(update=True)
    repo.index.commit(user_branch_name)
    username = credentials["username"]
    token = credentials["token"]
    try:
        repo.delete_remote("origin")
    except:
        pass
    origin_url = (
        f"https://{username}:{token}@github.com/{benchmark_owner}/{repo.name}.git"
    )
    origin = repo.create_remote("origin", url=origin_url)
    repo.git.push("--force", "--set-upstream", origin, repo.head.ref)
    # Tried this, but it did not work - returned an error
    """
    cmdline: git push -u origin test_user
    stderr: 'gh auth git-credential: "erase" operation not supported
    remote: Invalid username or password.
    """
    # origin = repo.remote("origin")
    # with repo.git.custom_environment(GIT_USERNAME=username, GIT_PASSWORD=token):
    #     repo.git.push("-u", "origin", "test_user")
    #     origin.push()
    commit_hash = repo.head.commit.hexsha
    return commit_hash


def get_repo(datapoint, repos_folder, test_username, benchmark_owner, credentials):
    """
    clones repo and switches it to the required commit
    creates branch with username
    """
    id = datapoint["id"]
    username = credentials["username"]
    token = credentials["token"]
    model_name = credentials["model"]
    repo_name, repo_owner = datapoint["repo_name"], datapoint["repo_owner"]
    # TODO add original branch name to new_branch_name
    new_branch_name = f"{test_username}__{model_name}__id_{id}"
    commit_hash = datapoint["sha_fail"]
    repo_path = os.path.join(repos_folder, f"{repo_owner}__{repo_name}")
    repo_url = f"https://github.com/{benchmark_owner}/{repo_name}.git"
    origin_url = (
        f"https://{username}:{token}@github.com/{benchmark_owner}/{repo_name}.git"
    )
    if (not os.path.exists(repo_path)) or (not os.listdir(repo_path)):
        repo = git.Repo.clone_from(repo_url, repo_path, depth=1)  # branch=commit_hash
    else:
        repo = git.Repo(repo_path)
    try:
        origin = repo.remote("origin")
    except:
        origin = repo.create_remote("origin", url=origin_url)
    repo.git.fetch("origin", commit_hash)
    try:
        repo.git.reset("--hard", commit_hash)
    except Exception as e:
        print(e)
        repo.git.checkout(commit_hash)
    # remove excessive files
    repo.git.clean("-fdx")
    if not any((h for h in repo.heads if h.name == new_branch_name)):
        # repo.delete_head("test_user", force=True)
        repo.create_head(new_branch_name, force=True)
    # TODO note that you should ban usage of the .git folder.
    # TODO discuss. May be store repos in the DPs
    # You need flag "-B" to checkout to the current state. Otherwise, the old brach state would be used
    repo.git.checkout("-B", new_branch_name)
    repo.name, repo.owner = repo_name, repo_owner

    return repo, new_branch_name


def get_run_data(repo_name, commit_sha, credentials, config):
    token = credentials["token"]
    headers = {"Authorization": f"token {token}"}

    jobs_url = f"https://api.github.com/repos/{config.benchmark_owner}/{repo_name}/commits/{commit_sha}/check-runs"
    response = requests.get(jobs_url, headers=headers)
    data = response.json()
    try:
        run_url = data["check_runs"][0]["html_url"]
        job_url = "/".join(run_url.split("/")[:-2])
        conclusions = [run["conclusion"] for run in data["check_runs"]]
        statuses = [run["status"] for run in data["check_runs"]]
        completed = [status == "completed" for status in statuses]
    except:
        print(f"Error in requesting jobs url {jobs_url}")
        print(data)
        job_url = ""
        conclusion = "error"
        return job_url, conclusion

    if not all(completed):
        conclusion = "waiting"
    elif "failure" in conclusions:
        conclusion = "failure"
    elif all([conclusion == "success" for conclusion in conclusions]):
        conclusion = "success"
    else:
        log_file_path = os.path.join(config.out_folder, "out_logs.txt")
        with open(log_file_path, "a") as f:
            f.write("--------------------DP BEGIN----------------------- \n")
            f.write(str(statuses) + "\n")
            f.write(str(conclusions) + "\n")
            f.write(str(data) + "\n")
            f.write("---------------------DP END------------------------- \n")
            conclusion = "error"

    return job_url, conclusion


def fix_none(datapoint, repo_path, repo=None, out_folder=None):
    return None


def fix_apply_diff(datapoint, repo_path, repo, out_folder):
    commit_sha = datapoint["sha_fail"][:7]
    diff_path = os.path.join(out_folder, f"{commit_sha}.diff")
    with open(diff_path, "w") as f:
        f.write(datapoint["diff"])
    try:
        repo.git.apply(diff_path)
    except GitCommandError as err:
        print(f"Sha = {datapoint['sha_fail']}")
        print(f"An error occurred while running the git command: {err}")
    os.remove(diff_path)
    return None


def process_datapoint(datapoint, fix_repo_function, config, credentials):
    """
    fix_repo_function - function that takes repo path and datapoint, repo object and out_folder.
    it should edit the repo in the folder, nothing to return
    credentials are passed in the following format:
    {'token': token, 'username': username}
    """

    # TODO think, what to do if test_username (which converts to a branch) is already present
    repo, user_branch_name = get_repo(
        datapoint,
        config.repos_folder,
        config.test_username,
        config.benchmark_owner,
        credentials,
    )
    # Prepares workflow file Moves target workflow file to the .github/workflows
    copy_and_edit_workflow_file(datapoint, repo)
    # Fixing the repo. fix_repo_function is provided by user.
    fix_repo_function(datapoint, repo.working_dir, repo, config.out_folder)
    # Push the corrected repo
    commit_sha = push_repo(repo, credentials, config.benchmark_owner, user_branch_name)
    job_identificator = {
        "repo_name": repo.name,
        "commit": commit_sha,
        "id": datapoint["id"],
        "sha_original": datapoint["sha_fail"],
        "branch_name": user_branch_name,
        "difficulty": datapoint["difficulty"],
    }

    return job_identificator


def get_results(job_identificator, config, credentials):
    # We have to make some pause to get result or even url, unless it sees no runs
    repo_name = job_identificator["repo_name"]
    commit_sha = job_identificator["commit"]
    job_url, conclusion = get_run_data(repo_name, commit_sha, credentials, config)

    return job_url, conclusion


def dataset_to_json(dataset):
    json_list = []
    for item in dataset:
        json_list.append(item)

    return json_list
