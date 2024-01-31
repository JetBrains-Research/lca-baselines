from tree_sitter import Language, Parser

PY_LANGUAGE = Language("build/languages.so", "python")
parser = Parser()
parser.set_language(Language("build/languages.so", "python"))
