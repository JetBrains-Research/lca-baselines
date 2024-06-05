import pytest

from src.data.preprocessing.parse_linked_issues import parse_linked_issues_from_comment, \
    parse_linked_issues_from_comments
from tests import TEST_ROOT_PATH
from src.utils.jsonl_utils import save_jsonl_data


@pytest.mark.parametrize(
    "comment_body, linked_issues",
    [
        ("Solves https://github.com/jlord/sheetsee.js/issues/263", [(263, "solves", "issue_link")]),
        ("Fixes #262", [(262, "fixes", "hash")]),
        ("Hey!\nResolves GH-264 and GH-265, GH-268 fixed. Also Fixes #262.",
         [(262, "fixes", "hash"), (264, "resolves", "slash"), (265, "", "slash"), (268, "fixed", "slash")]),
        # Double parsing, but ok
        ("Resolved jlord/sheetsee.js#263 fixed", [(263, "fixed", "hash"), (263, "resolved", "file")]),
    ],
)
def test_parse_linked_issues_from_comment(comment_body: str, linked_issues: list[str]):
    parsed_linked_issues = parse_linked_issues_from_comment(comment_body)

    assert parsed_linked_issues == linked_issues


@pytest.mark.parametrize(
    "repo, linked_issues",
    [
        (
                {"owner": "owner0", "name": "name0"},
                [
                    {
                        "comment_html_url": "https://github.com/owner0/name0/issue/2#issuecomment-2",
                        "issue_html_url": "https://github.com/owner0/name0/issue/2",
                        "linked_issue_html_url": "https://github.com/owner0/name0/issues/1",
                        "link_type": "issue_link",
                    },
                    {
                        "comment_html_url": "https://github.com/owner0/name0/pull/5#issuecomment-2",
                        "issue_html_url": "https://github.com/owner0/name0/pull/5",
                        "linked_issue_html_url": "https://github.com/owner0/name0/issues/2",
                        "link_type": "issue_link",
                    },
                    {
                        "comment_html_url": "https://github.com/owner0/name0/pull/6#issuecomment-3",
                        "issue_html_url": "https://github.com/owner0/name0/pull/6",
                        "linked_issue_html_url": "https://github.com/owner0/name0/issues/3",
                        "link_type": "file",
                    },
                ]
        ),
        (
                {"owner": "owner1", "name": "name1"},
                [
                    {
                        "comment_html_url": "https://github.com/owner1/name1/issue/3#issuecomment-2",
                        "issue_html_url": "https://github.com/owner1/name1/issue/3",
                        "linked_issue_html_url": "https://github.com/owner1/name1/issues/1",
                        "link_type": "hash",
                    },
                    {
                        "comment_html_url": "https://github.com/owner1/name1/pull/6#issuecomment-4",
                        "issue_html_url": "https://github.com/owner1/name1/pull/6",
                        "linked_issue_html_url": "https://github.com/owner1/name1/issues/1",
                        "link_type": "issue_link",
                    },
                    {
                        "comment_html_url": "https://github.com/owner1/name1/pull/7#issuecomment-3",
                        "issue_html_url": "https://github.com/owner1/name1/pull/7",
                        "linked_issue_html_url": "https://github.com/owner1/name1/issues/2",
                        "link_type": "hash",
                    },
                    {
                        "comment_html_url": "https://github.com/owner1/name1/pull/7#issuecomment-3",
                        "issue_html_url": "https://github.com/owner1/name1/pull/7",
                        "linked_issue_html_url": "https://github.com/owner1/name1/issues/1",
                        "link_type": "slash",
                    },
                    {
                        "comment_html_url": "https://github.com/owner1/name1/pull/7#issuecomment-3",
                        "issue_html_url": "https://github.com/owner1/name1/pull/7",
                        "linked_issue_html_url": "https://github.com/owner1/name1/issues/3",
                        "link_type": "slash",
                    },
                ]
        ),
    ],
)
def test_parse_linked_issues_from_comments(repo: dict, linked_issues: list[dict]):
    parsed_linked_issues = parse_linked_issues_from_comments(
        repo['owner'], repo['name'], TEST_ROOT_PATH / "resources" / "comments",
                                     TEST_ROOT_PATH / "resources" / "comments",
    )

    save_jsonl_data(
        repo['owner'], repo['name'],
        parsed_linked_issues,
        TEST_ROOT_PATH / "resources" / "issues_links",
    )

    assert parsed_linked_issues == linked_issues
