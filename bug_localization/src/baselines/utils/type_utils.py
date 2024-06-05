from typing import TypedDict


class ChatMessage(TypedDict):
    role: str
    content: str
