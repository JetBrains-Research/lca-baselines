from typing import List, Dict

from src.baselines.backbones.agent.env.fs_tools import read_fs_tools


class FileSystemEnv:

    def __init__(self, repo_content: Dict[str, str]):
        self.repo_content = repo_content

    def _read_file(self, path: str) -> str:
        if path in self.repo_content:
            return self.repo_content[path]
        else:
            return f"Error occurred while reading file. Repository does not contain file {path}."

    def _list_directory(self, path: str) -> List[str]:
        if path == "" or path == ".":
            files_by_path = list(self.repo_content.keys())
            return list(set(f.split('/', 1)[0] for f in files_by_path))
        dir_path = path + '/'
        files_by_path = [f for f in self.repo_content.keys() if f.startswith(dir_path)]
        return list(set([dir_path + f.replace(dir_path, "").split('/', 1)[0] for f in files_by_path]))

    def _assert_args(self, command_name: str, command_params, expected_args: List[str]):
        for arg in expected_args:
            assert command_params.get(arg) is not None, Exception(f"Argument {arg} is not provided for tool call {command_name}")

    def run_command(self, command_name: str, command_params: dict) -> str:
        try:
            message = ""
            if command_name == 'read_file':
                self._assert_args(command_name, command_params, ['path'])
                message = self._read_file(
                    path=command_params.get("path"),
                )
            elif command_name == 'list_directory':
                self._assert_args(command_name, command_params, ['path'])
                message = str(self._list_directory(
                    path=command_params.get("path"),
                ))
            return message
        except Exception as e:
            return str(e)

    def get_tools(self) -> list[dict]:
        return read_fs_tools
