import os

from composers.one_completion_file_composer import OneCompletionFileComposer
from data_classes.datapoint_base import DatapointBase


class PathDistanceComposer(OneCompletionFileComposer):
    @staticmethod
    def _path_distance(path_from, path_to):
        divided_path_from = os.path.normpath(path_from).split(os.path.sep)
        divided_path_to = os.path.normpath(path_to).split(os.path.sep)
        common_len = 0
        for el1, el2 in zip(divided_path_from, divided_path_to):
            if el1 == el2:
                common_len += 1
            else:
                break
        # return len(divided_path_from) - common_len - 1
        return (len(divided_path_from) - common_len - 1) + (len(divided_path_to) - common_len - 1)

    def _sort_filepathes(self, path_from, list_of_filepathes):
        max_len = max([len(os.path.normpath(path).split(os.path.sep)) for path in list_of_filepathes])
        max_len += len(os.path.normpath(path_from).split(os.path.sep))
        paths_by_distance = [list() for _ in range(max_len)]

        for path_to in list_of_filepathes:
            dist = self._path_distance(path_from, path_to)
            paths_by_distance[dist].append(path_to)
        return [path for path_group in paths_by_distance for path in path_group]

    def context_composer(self, datapoint: DatapointBase) -> str:
        # context = datapoint['context']
        context = datapoint.get_context()
        completion = datapoint.get_completion()
        repo_name = datapoint.repo_name
        assert len(completion) == 1, 'Only one file should be completed'
        completion_path = list(completion)[0]
        sorted_pathes = self._sort_filepathes(completion_path, list(context))
        composed_content = [path + self.meta_info_sep_symbol + context[path] for path in sorted_pathes[::-1]]

        composed_content.append(completion_path + self.meta_info_sep_symbol)

        repo_metainfo = f"{self.extension}{self.lang_sep_symbol}{repo_name}{self.meta_info_sep_symbol}"

        return repo_metainfo + self.lang_sep_symbol.join(composed_content)
