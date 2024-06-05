from composers.one_completion_file_composer import OneCompletionFileComposer
from data_classes.datapoint_base import DatapointBase


class FileLengthComposer(OneCompletionFileComposer):
    @staticmethod
    def _get_filelengths(context: dict[str, str]) -> dict[int, str]:
        set_of_len = set([len(file) for file in context.values()])
        if len(context) == len(set_of_len):
            len_to_path = {len(file): path for path, file in context.items()}
            return len_to_path
        else:
            len_to_path = dict()
            for path, file in context.items():
                if len(file) not in len_to_path.keys():
                    len_to_path[len(file)] = path
                else:
                    new_key = len(file) + 1
                    while new_key in len_to_path.keys():
                        new_key += 1
                    len_to_path[new_key] = path
            return len_to_path

    def context_composer(self, datapoint: DatapointBase) -> str:
        # context = datapoint['context']
        context = datapoint.get_context()
        len_to_path = self._get_filelengths(context)

        completion = datapoint.get_completion()
        repo_name = datapoint.repo_name
        assert len(completion) == 1, 'Only one file should be completed'
        completion_path = list(completion)[0]
        composed_content = [path + self.meta_info_sep_symbol + context[path] for _, path in
                            sorted(len_to_path.items(), reverse=True)]

        composed_content.append(completion_path + self.meta_info_sep_symbol)

        repo_metainfo = f"{self.extension}{self.lang_sep_symbol}{repo_name}{self.meta_info_sep_symbol}"

        return repo_metainfo + self.lang_sep_symbol.join(composed_content)
