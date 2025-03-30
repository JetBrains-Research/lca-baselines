import re
from typing import List, Tuple

from rank_bm25 import BM25Okapi


def tokenize_with_regex(text: str) -> List[str]:
    """
    Tokenizes the input text using a simple regex.
    Separates words, numbers, and punctuation as tokens.
    """
    token_pattern = r"\w+|[^\w\s]"  # Match words, numbers, or punctuation
    return re.findall(token_pattern, text)


def tokenize_with_tiktoken(text: str, model_name: str) -> List[str]:
    """
    Tokenizes the input text using tiktoken.
    Encodes text and converts tokens to strings to match BM25 format.
    """
    import tiktoken  # Importing here so it's optional if using regex
    enc = tiktoken.encoding_for_model(model_name)

    return [str(token) for token in enc.encode(text)]


def sort_files_by_relevance(
        repo_content: dict,
        issue_text: str,
        tokenizer_name: str = "tiktoken",
        model_name: str = "gpt-4o"
) -> List[Tuple[str, str]]:
    if tokenizer_name == "regex":
        tokenizer = lambda text: tokenize_with_regex(text)
    elif tokenizer_name == "tiktoken":
        tokenizer = lambda text: tokenize_with_tiktoken(text, model_name)
    else:
        raise Exception("Unsupported tokenizer")

    # Tokenize the issue text
    issue_tokens = tokenizer(issue_text)

    # Tokenize file contents
    tokenized_files = [
        (path, tokenizer(path + '\n' + content))
        for path, content in repo_content.items()
    ]

    # Prepare the BM25 model with tokenized file contents
    bm25 = BM25Okapi([tokens for _, tokens in tokenized_files])

    # Get BM25 scores for the issue tokens against all files
    scores = bm25.get_scores(issue_tokens)

    # Combine file paths and scores, then sort by highest score
    ranked_files = sorted(
        zip(repo_content.items(), scores),
        key=lambda x: x[1],
        reverse=True
    )

    # Return sorted file paths with their original contents
    return [(file[0], file[1]) for file, _ in ranked_files]


if __name__ == '__main__':
    # Example usage with regex tokenizer
    print("Using regex tokenizer:")
    print("print(b + c) line error")
    print(sort_files_by_relevance(
        repo_content={
            "a.py": "so irrelevant code just to check sorting works",
            "b.py": "print(b + c)",
            "c.py": "print(a + b)",
        },
        issue_text="print(b + c) line error",
        tokenizer_name='regex'
    ))

    # Example usage with tiktoken tokenizer
    print("Using tiktoken tokenizer:")
    print("print(b + c) line error")
    print(sort_files_by_relevance(
        repo_content={
            "a.py": "so irrelevant code just to check sorting works",
            "b.py": "print(b + c)",
            "c.py": "print(a + b)",
        },
        issue_text="print(b + c) line error",
        tokenizer_name='tiktoken'
    ))
