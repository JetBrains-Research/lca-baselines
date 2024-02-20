# Directory content description

## [Parse Linked Issues](parse_linked_issues.py)

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
