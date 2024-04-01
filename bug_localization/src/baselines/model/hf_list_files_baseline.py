import os
from typing import Any, Dict

from dotenv import load_dotenv
from omegaconf import OmegaConf
from tenacity import wait_random_exponential, stop_after_attempt, retry
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import pipeline

from src.baselines.model.baseline import Baseline
from src.utils.git_utils import get_repo_content_on_commit
from src.utils.hf_utils import load_data
from src.utils.tokenization_utils import TokenizationUtils


class HFListFilesBaseline(Baseline):

    def __init__(self, model: str, profile: str, max_tokens: int):
        self.model = model
        self.profile = profile
        self.tokenizer = AutoTokenizer.from_pretrained(self.model)
        self.model = AutoModelForCausalLM.from_pretrained(self.model)

    @staticmethod
    def name():
        return 'hf'

    @staticmethod
    def _build_messages(issue_text: str, file_paths: list[str]) -> str:
        file_paths_list = '\n'.join(file_paths)

        return f"""List of files:\n {file_paths_list} + '\n' + 
                Issue: \n {issue_text} \n
                You are given a list of files in project.
                Select subset of them which should be fixed according to issue.
                As a response provide ONLY a list of files separated with line separator
                without any comments.
                """

    def _batch_project_files(self, issue_description: str, file_paths: list[str]) -> list[list[str]]:
        tokenization_utils = TokenizationUtils(self.profile)

        has_big_message = True
        n = len(file_paths)
        step = len(file_paths)

        while has_big_message:
            has_big_message = False
            for i in range(0, n, step):
                message = self._build_messages(issue_description, file_paths[i:i + step])
                if tokenization_utils.count_text_tokens(message) > self.max_tokens:
                    has_big_message = True
                    step //= 2
                    break

        batched_files_path = [file_paths[i:i + step] for i in range(0, n, step)]
        assert len(file_paths) == sum(len(f) for f in batched_files_path)
        print(len(batched_files_path))
        return [file_paths[i:i + step] for i in range(0, n, step)]

    @retry(wait=wait_random_exponential(multiplier=1, max=40), stop=stop_after_attempt(3))
    def localize_bugs(self, issue_description: str, repo_content: dict[str, str]) -> Dict[str, Any]:
        all_project_files = list(repo_content.keys())
        batched_project_files = self._batch_project_files(issue_description, all_project_files)

        expected_files = []
        for project_files in batched_project_files:
            message = self._build_messages(issue_description, project_files)
            code_generator = pipeline('text-generation', model=self.model, tokenizer=self.tokenizer)
            generated_code = code_generator(message, max_length=1000)[0]['generated_text']

            for file in generated_code.split('\n'):
                if file in project_files:
                    expected_files.append(file)

        return {
            "expected_files": expected_files
        }


def main():
    load_dotenv()
    baseline = HFListFilesBaseline("codellama/CodeLlama-7b-Instruct-hf", 'chat-llama-v2-7b')
    config = OmegaConf.load("../../../configs/data/server.yaml")

    df = load_data('py', 'test')
    for dp in df:
        repo_path = os.path.join(config.repos_path, f"{dp['repo_owner']}__{dp['repo_name']}")
        repo_content = get_repo_content_on_commit(repo_path, dp['base_sha'], ['py'])
        result = baseline.localize_bugs(
            dp['issue_body'],
            repo_content,
        )
        print(result)
        print(dp['changed_files'])

        return


if __name__ == '__main__':
    main()
