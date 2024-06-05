import os

from composers.one_completion_file_composer import OneCompletionFileComposer
from data_classes.datapoint_base import DatapointBase
from tree_sitter_parser.parsed_file import ParsedFile


class ImportsFirstComposer(OneCompletionFileComposer):
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
