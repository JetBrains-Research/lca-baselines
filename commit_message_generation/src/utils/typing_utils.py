from typing import List, Optional

from typing_extensions import TypedDict


class ChatMessage(TypedDict):
    role: str
    content: str


class CommitDiff(TypedDict):
    change_type: str
    new_path: Optional[str]
    old_path: Optional[str]
    diff: str


class CommitFile(TypedDict):
    old_path: str
    new_path: str
    old_contents: str
    new_contents: str


class CommitMods(TypedDict):
    diff: Optional[List[CommitDiff]]
    file_src: Optional[CommitFile]


class UnifiedCommitExample(TypedDict):
    hash: str
    repo: str
    date: str
    license: str
    message: str
    mods: CommitMods
