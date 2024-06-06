from composers.one_completion_file_composer import OneCompletionFileComposer
from data_classes.datapoint_base import DatapointBase


class HalfMemoryComposer(OneCompletionFileComposer):
    @staticmethod
    def _forget_half(file_content: str) -> str:
        from random import random
        result = list()
        for line in file_content.split('\n'):
            if random() > 0.5:
                result.append(line)
        return "\n".join(result)

    def context_composer(self, datapoint: DatapointBase) -> str:
        # context = datapoint['context']
        context = datapoint.get_context()
        completion = datapoint.get_completion()
        repo_name = datapoint.repo_name
        assert len(completion) == 1, 'Only one file should be completed'
        completion_path = list(completion)[0]

        composed_content = [path + self.meta_info_sep_symbol + self._forget_half(content) for path, content in context.items()]

        composed_content.append(completion_path + self.meta_info_sep_symbol)

        repo_metainfo = f"{self.extension}{self.lang_sep_symbol}{repo_name}{self.meta_info_sep_symbol}"

        return repo_metainfo + self.lang_sep_symbol.join(composed_content)
