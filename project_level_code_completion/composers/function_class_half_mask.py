from composers.path_distance_composer import PathDistanceComposer
from data_classes.datapoint_base import DatapointBase


class FuncClassComposer(PathDistanceComposer):
    @staticmethod
    def _filter_func_class(code: str) -> str:
        lines = code.split('\n')
        filtered_lines = lines.copy()
        current_mask = None
        for idx, line in enumerate(lines):
            if not (line.strip().startswith('def') or line.strip().startswith('class')):
                filtered_lines[idx] = current_mask
                if current_mask:
                    current_mask = None
            else:
                current_mask = 'pass'
        filtered_lines = [l for l in filtered_lines if l]
        return '\n'.join(filtered_lines)

    def context_composer(self, datapoint: DatapointBase) -> str:
        # context = datapoint['context']
        context = datapoint.get_context()
        completion = datapoint.get_completion()
        repo_name = datapoint.repo_name
        assert len(completion) == 1, 'Only one file should be completed'
        completion_path = list(completion)[0]
        sorted_pathes = self._sort_filepathes(completion_path, list(context))

        composed_content = list()
        for file_num, path in enumerate(sorted_pathes[::-1]):
            if file_num < len(sorted_pathes) * 0.5:
                code = self._filter_func_class(context[path])
            else:
                code = context[path]

            composed_content.append(path + self.meta_info_sep_symbol + code)

        # composed_content = [path + self.meta_info_sep_symbol + context[path] for path in sorted_pathes[::-1]]

        composed_content.append(completion_path + self.meta_info_sep_symbol)

        repo_metainfo = f"{self.extension}{self.lang_sep_symbol}{repo_name}{self.meta_info_sep_symbol}"

        return repo_metainfo + self.lang_sep_symbol.join(composed_content)

