import os
from itertools import chain

from code_completion.tree_sitter_parser.parsed_file import ParsedFile


class ParsedProject:

    def __init__(self, project_root: str, skip_directories: list[str] = None):
        if skip_directories is None:
            skip_directories = []

        self.parsed_files: list[ParsedFile] = []

        for project_subroot in os.listdir(project_root):
            project_subroot = os.path.join(project_root, project_subroot)

            if "example" in project_subroot.lower() or not os.path.isdir(project_subroot):
                continue

            for dirpath, dirnames, filenames in os.walk(project_subroot):
                for filename in filenames:
                    if filename.endswith(".py"):
                        filepath = os.path.join(dirpath, filename)
                        parsed_file = ParsedFile(filepath)
                        self.parsed_files.append(parsed_file)

        self.defined_functions = set(chain.from_iterable(
            parsed_file.function_names
            for parsed_file in self.parsed_files
        ))

        self.defined_classes = set(chain.from_iterable(
            parsed_file.class_names
            for parsed_file in self.parsed_files
        ))
