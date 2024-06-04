import json
from typing import Dict, Any
import os

from data_classes.datapoint_base import DatapointBase
from tree_sitter_parser.parsed_file import ParsedFile

COMPOSERS = {
    'none': None,
    'naive': {'module': 'eval.composers', 'name': 'DummyComposer'},
    # 'alphabetical': {'module': 'lca.code_generation.eval.composers', 'name': 'AlphabeticalComposer'},
    'path_distance': {'module': 'eval.composers', 'name': 'PathDistanceComposer'},
    # 'file_length': {'module': 'eval.composers', 'name': 'FileLengthComposer'},
    'half_memory': {'module': 'eval.composers', 'name': 'HalfMemoryComposer'},
    'half_memory_path': {'module': 'eval.composers', 'name': 'HalfMemoryPathComposer'},
    # 'function_class_mask_half': {'module': 'eval.composers', 'name': 'FuncClassComposer'},
    'function_class_mask_one': {'module': 'eval.composers', 'name': 'FuncClassComposerOne'},
    # 'imports_first': {'module': 'eval.composers', 'name': 'ImportsFirstComposer'},
}


class ComposerBase:
    def __init__(self,
                 lang_sep_symbol,
                 meta_info_sep_symbol,
                 extension,
                 ):
        self.lang_sep_symbol = lang_sep_symbol
        self.meta_info_sep_symbol = meta_info_sep_symbol
        self.extension = extension
    
    def context_composer(self, datapoint: DatapointBase) -> str:
        raise NotImplementedError

    def completion_composer(self, datapoint: DatapointBase) -> str:
        raise NotImplementedError


class OneCompletonFileComposer(ComposerBase):
    def __init__(self, **composer_args):
        super().__init__(**composer_args)

    def completion_composer(self, datapoint: DatapointBase) -> str:
        completion = datapoint.completion_dict
        assert len(completion) == 1, 'Only one file should be completed'
        content = list(completion.values())[0]
        return content


class DummyComposer(OneCompletonFileComposer):
    def context_composer(self, datapoint: DatapointBase) -> str:
        # context = datapoint['context']
        context = datapoint.get_context()
        repo_name = datapoint.repo_name
        composed_content = [path + self.meta_info_sep_symbol + content for path, content in context.items()]

        completion = datapoint.get_completion()
        assert len(completion) == 1, 'Only one file should be completed'
        completion_path = list(completion)[0]
        composed_content.append(completion_path + self.meta_info_sep_symbol)

        repo_metainfo = f"{self.extension}{self.lang_sep_symbol}{repo_name}{self.meta_info_sep_symbol}"

        return repo_metainfo + self.lang_sep_symbol.join(composed_content)


    

class AlphabeticalComposer(OneCompletonFileComposer):
    def context_composer(self, datapoint: DatapointBase) -> str:
        # context = datapoint['context']
        context = datapoint.get_context()
        repo_name = datapoint.repo_name
        composed_content = [path + self.meta_info_sep_symbol + content for path, content in sorted(context.items())]

        completion = datapoint.get_completion()
        assert len(completion) == 1, 'Only one file should be completed'
        completion_path = list(completion)[0]
        composed_content.append(completion_path + self.meta_info_sep_symbol)

        repo_metainfo = f"{self.extension}{self.lang_sep_symbol}{repo_name}{self.meta_info_sep_symbol}"

        return repo_metainfo + self.lang_sep_symbol.join(composed_content)


class PathDistanceComposer(OneCompletonFileComposer):
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


class FileLengthComposer(OneCompletonFileComposer):
    @staticmethod
    def _get_filelengths(context: dict[str, str]) -> Dict[int, str]:
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


class HalfMemoryComposer(OneCompletonFileComposer):
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


class HalfMemoryPathComposer(PathDistanceComposer, HalfMemoryComposer):
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


class FuncClassComposerOne(FuncClassComposer):
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
            if file_num + 1.5 < len(sorted_pathes):
                code = self._filter_func_class(context[path])
            else:
                code = context[path]

            composed_content.append(path + self.meta_info_sep_symbol + code)

        # composed_content = [path + self.meta_info_sep_symbol + context[path] for path in sorted_pathes[::-1]]

        composed_content.append(completion_path + self.meta_info_sep_symbol)

        repo_metainfo = f"{self.extension}{self.lang_sep_symbol}{repo_name}{self.meta_info_sep_symbol}"

        return repo_metainfo + self.lang_sep_symbol.join(composed_content)


class ImportsFirstComposer(OneCompletonFileComposer):
    imports_query = """
        (import_statement 
              (dotted_name) @module_name)
        (import_from_statement 
              (dotted_name) @module_from_name)
        (import_statement 
              (aliased_import) @module_name)
        (import_from_statement 
              (aliased_import) @module_from_name)
    """

    def _sort_imports(self, datapoint: DatapointBase, context: dict[str, str]) -> list[str]:
        def get_imported_modules(code: str) -> set[str]:
            pf = ParsedFile(code=code)
            bmodules = pf.make_query(self.imports_query)
            return {m.decode(pf.encoding) for m in bmodules if b'.' in m}

        def filepath_to_module(filepath:str) -> str:
            return os.path.splitext(filepath)[0].replace(os.path.sep, '.').lstrip('.')

        def is_file_imported(filepath: str, code: str) -> bool:
            imported_modules = get_imported_modules(code)
            target_module = filepath_to_module(filepath)
            return (any([m in target_module for m in imported_modules]) or
                    any([target_module in m for m in imported_modules]))

        code = list(datapoint.get_completion().values())[0]
        # imported_files = [fp for fp in datapoint['context'] if is_file_imported(fp, code)]
        # notimported_files = [fp for fp in datapoint['context'] if not is_file_imported(fp, code)]
        imported_files = [fp for fp in context if is_file_imported(fp, code)]
        nonimported_files = [fp for fp in context if not is_file_imported(fp, code)]

        return nonimported_files + imported_files

    def context_composer(self, datapoint: DatapointBase) -> str:
        # context = datapoint['context']
        context = datapoint.get_context()
        completion = datapoint.get_completion()
        repo_name = datapoint.repo_name
        assert len(completion) == 1, 'Only one file should be completed'
        completion_path = list(completion)[0]

        composed_content = [path + self.meta_info_sep_symbol + context[path] for path in
                            self._sort_imports(datapoint, context)]

        composed_content.append(completion_path + self.meta_info_sep_symbol)

        repo_metainfo = f"{self.extension}{self.lang_sep_symbol}{repo_name}{self.meta_info_sep_symbol}"

        return repo_metainfo + self.lang_sep_symbol.join(composed_content)




# datapoint = {
#     'repo_name': 'name',
#     'context': {'fp1': 'a\nb\ndef c\nd\nclass e', 'fp2': 'def a\nb\nc\nd\ne', 'fp3': 'a\ndef b\nc\nd\ne', 'fp4': 'a\nb\nclass c\nd\ne',},
#     'completion': {'re': 'uyiyo\nyututi\n'}
# }
# comp = ImportsFirstComposer(lang_sep_symbol='L\n',
#                  meta_info_sep_symbol='M\n',
#                  extension='.RU\n',)

# print(comp.context_composer(datapoint))
# dataset_path = "/home/glukhov/long_code_arena/lca/data/python/benchmark_data_smol.json"
# with open(dataset_path, "r") as f:
#     data = json.load(f)
#
# print(list(data[0]))
#
# for datapoint in data:
#     datapoint['context'] = {k: "er\n" for k in datapoint['context'].keys()}
#     datapoint['completion'] = {k: v[:200] for k, v in datapoint['completion'].items()}
#     print(comp.context_composer(datapoint))
#     print(comp.completion_composer(datapoint))
#     print("="*100)
