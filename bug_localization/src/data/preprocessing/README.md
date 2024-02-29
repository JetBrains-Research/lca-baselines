# Scripts for data preprocessing

### [parse_linked_issues.py](parse_linked_issues.py)

Gets all issues and pull requests comments contents and parse all issues links from them
in format jsonl `{repo_owner}__{repo_name}.jsonl` with list of jsons in format
```json
{
  "comment_html_url": "https://github.com/AmadeusITGroup/otter/pull/63#issuecomment-1416400069",
  "issue_html_url": "https://github.com/AmadeusITGroup/otter/pull/63",
  "linked_issue_html_url": "https://github.com/AmadeusITGroup/otter/issues/25",
  "link_type": "hash"
}
```
where
* `comment_html_url` -- url of the comment from where link was parsed
* `issue_html_url` -- url of the issue from where link was parsed
* `linked_issue_html_url` -- url of the issue where link leads
* `link_type` -- type of issue linkage

Jsons are saved to `issues_links_path` defined in [config](../../../configs/data/server.yaml).


### [filter_linked_issues.py](filter_linked_issues.py)
Gets all issues <-> linked issues links and leaves only:
* issue <-> pull request links
* issue/pull requests with additional loaded info
* able to get diff and repo initial state from pull request
* diff has at least one of py|java|kt file
* issue text has no media content
* issue text has utf-8 encoding

in format jsonl `{repo_owner}__{repo_name}.jsonl`. 
Jsons are saved to `issues_links_filtered_path` defined in [config](../../../configs/data/server.yaml).

### [prepare_data_for_baseline.py](prepare_data_for_baseline.py)
Collects all gathered data to jsonl/csv files splited by language:
* py - diff contains files written on Python
* java - diff contains files written on Java
* kt - diff contains files written on Kotlin
* mixed - diff contains files written on Kotlin|Java|Python + other files like shell scripts etc.

jsonl contains following jsons:
```json
{
    "id": datasets.Value("int64"),
    "repo_owner": datasets.Value("string"),
    "repo_name": datasets.Value("string"),
    "issue_url": datasets.Value("string"),
    "pull_url": datasets.Value("string"),
    "comment_url": datasets.Value("string"),
    "links_count": datasets.Value("int64"),
    "issue_title": datasets.Value("string"),
    "issue_body": datasets.Value("string"),
    "base_sha": datasets.Value("string"),
    "head_sha": datasets.Value("string"),
    "diff_url": datasets.Value("string"),
    "diff": datasets.Value("string"),
    "changed_files": datasets.Value("string"),
    "changed_files_exts": datasets.Value("string"),
    "changed_files_count": datasets.Value("int64"),
    "java_changed_files_count": datasets.Value("int64"),
    "kt_changed_files_count": datasets.Value("int64"),
    "py_changed_files_count": datasets.Value("int64"),
    "code_changed_files_count": datasets.Value("int64"),
    "pull_create_at": datasets.Value("string"),
    "stars": datasets.Value("int64"),
    "language": datasets.Value("string"),
    "languages": datasets.Value("string"),
    "license": datasets.Value("string"),
}
```