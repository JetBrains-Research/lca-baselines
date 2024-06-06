from composers.half_memory_composer import HalfMemoryComposer
from composers.path_distance_composer import PathDistanceComposer
from data_classes.datapoint_base import DatapointBase


class HalfMemoryPathDistanceComposer(PathDistanceComposer, HalfMemoryComposer):
    def context_composer(self, datapoint: DatapointBase) -> str:
        # context = datapoint['context']
        context = datapoint.get_context()
        completion = datapoint.get_completion()
        repo_name = datapoint.repo_name
        assert len(completion) == 1, 'Only one file should be completed'
        completion_path = list(completion)[0]
        sorted_pathes = self._sort_filepathes(completion_path, list(context))

        composed_content = [path + self.meta_info_sep_symbol + self._forget_half(context[path]) for path in sorted_pathes[::-1]]

        composed_content.append(completion_path + self.meta_info_sep_symbol)

        repo_metainfo = f"{self.extension}{self.lang_sep_symbol}{repo_name}{self.meta_info_sep_symbol}"

        return repo_metainfo + self.lang_sep_symbol.join(composed_content)
