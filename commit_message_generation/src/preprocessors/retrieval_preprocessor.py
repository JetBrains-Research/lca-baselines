import os
from pathlib import Path
from typing import Any, Dict

from langchain_community.retrievers import BM25Retriever

from ..retrieval.hf_document_loader import HuggingFaceGitReposLoader
from ..utils import TokenizationUtils
from ..utils.typing_utils import UnifiedCommitExample
from . import SimpleCMGPreprocessor


class RetrievalCMGPreprocessor(SimpleCMGPreprocessor):
    """Concatenates all file diffs into a single diff."""

    def __init__(
        self,
        model_name: str,
        model_provider: str,
        max_num_tokens: int,
        hf_repo_id: str,
        hf_path_in_repo: str,
        local_data_dir: str,
        include_path: bool = True,
        *args,
        **kwargs,
    ):
        super().__init__(
            model_name=model_name,
            model_provider=model_provider,
            include_path=include_path,
        )
        self.hf_repo_id = hf_repo_id
        self.hf_path_in_repo = hf_path_in_repo
        self.local_data_dir = local_data_dir
        self.max_num_tokens = max_num_tokens
        self._tokenization_utils = TokenizationUtils(model_name=model_name, model_provider=model_provider)

    def __call__(self, commit: UnifiedCommitExample, **kwargs) -> Dict[str, Any]:
        processed_commit_mods = super().__call__(commit)["mods"]

        document_loader = HuggingFaceGitReposLoader(
            hf_repo_id=self.hf_repo_id,
            hf_path_in_repo=self.hf_path_in_repo,
            local_data_dir=self.local_data_dir,
            repository=commit["repo"],
            hash=commit["hash"],
            files_to_exclude={mod["old_path"] for mod in commit["mods"]},
        )
        docs = document_loader.load()
        retriever = BM25Retriever.from_documents(docs, k=50)
        context = retriever.invoke(processed_commit_mods)

        resulting_context = []
        cur_num_tokens = self._tokenization_utils.count_tokens(processed_commit_mods)
        for doc in context:
            if "source" in doc.metadata:
                path = Path(doc.metadata["source"])
                relative_path = str(
                    path.relative_to(
                        Path(os.path.join(document_loader.extracted_dir, commit["repo"].replace("/", "__")))
                    )
                )
                doc.metadata["source"] = relative_path

            num_tokens = self._tokenization_utils.count_tokens(doc.page_content)
            if num_tokens + cur_num_tokens > self.max_num_tokens:
                trunctated_doc = (
                    self._tokenization_utils.truncate(
                        doc.page_content, max_num_tokens=self.max_num_tokens - cur_num_tokens
                    )
                    + "\n\n[... the rest is omitted ...]"
                )
                resulting_context.append({**doc.metadata, "content": trunctated_doc})
                cur_num_tokens += self._tokenization_utils.count_tokens(trunctated_doc)
                break
            else:
                resulting_context.append({**doc.metadata, "content": doc.page_content})
                cur_num_tokens += num_tokens

        return {"mods": processed_commit_mods, "context": resulting_context}
