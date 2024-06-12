from composers.one_completion_file_composer import OneCompletionFileComposer
from data_classes.datapoint_base import DatapointBase


class AlphabeticalComposer(OneCompletionFileComposer):
    def context_composer(self, datapoint: DatapointBase) -> str:
        # context = datapoint['context']
        context = datapoint.get_context()
        repo_name = datapoint.repo_name
        composed_content = [
            path + self.meta_info_sep_symbol + content for path, content in sorted(context.items(), reverse=True)
        ]

        completion = datapoint.get_completion()
        assert len(completion) == 1, 'Only one file should be completed'
        completion_path = list(completion)[0]
        composed_content.append(completion_path + self.meta_info_sep_symbol)

        repo_metainfo = f"{self.extension}{self.lang_sep_symbol}{repo_name}{self.meta_info_sep_symbol}"

        return repo_metainfo + self.lang_sep_symbol.join(composed_content)
