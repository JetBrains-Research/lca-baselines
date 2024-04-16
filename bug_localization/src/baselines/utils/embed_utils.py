from typing import Dict

import numpy as np


def data_to_vectors(issue_description: str, repo_content: Dict[str, str]) \
        -> tuple[np.ndarray[str], np.ndarray[str]]:

    issue_text = issue_description

    file_names = ["issue_text"]
    file_contents = [issue_text]
    for file_name, file_content in repo_content.items():
        file_names.append(file_name)
        file_contents.append(file_name + "\n" + file_content)

    return (np.asarray(file_names, dtype=str), np.asarray(file_contents, dtype=str))
