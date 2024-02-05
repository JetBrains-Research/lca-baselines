from tree_sitter import Language, Parser

try:
    PY_LANGUAGE = Language("build/languages.so", "python")
except OSError:
    Language.build_library("build/languages.so", ["tree-sitter-python"])
    PY_LANGUAGE = Language("build/languages.so", "python")


parser = Parser()
parser.set_language(Language("build/languages.so", "python"))
