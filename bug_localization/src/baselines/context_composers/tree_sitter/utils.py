from pathlib import Path

from tree_sitter import Language, Parser

BASE_DIR = Path(__file__).parent.resolve()

# Build the Java language library
Language.build_library(
    BASE_DIR / "tree-sitter-java.so",  # Output file
    [BASE_DIR / "tree-sitter-java"]  # Source folder
)

JAVA_LANGUAGE = Language(BASE_DIR / "tree-sitter-java.so", "java")

# Build the Kotlin language library
Language.build_library(
    BASE_DIR / "tree-sitter-kotlin.so",  # Output file
    [BASE_DIR / "tree-sitter-kotlin"]  # Source folder
)
KOTLIN_LANGUAGE = Language(BASE_DIR / "tree-sitter-kotlin.so", "kotlin")

# Build the Python language library
Language.build_library(
    BASE_DIR / "tree-sitter-python.so",  # Output file
    [BASE_DIR / "tree-sitter-python"]  # Source folder
)
PYTHON_LANGUAGE = Language(BASE_DIR / "tree-sitter-python.so", "python")


def get_parser(language):
    parser = Parser()
    parser.set_language(language)
    return parser


# Extraction logic for Python
def extract_python_imports(source_code):
    parser = get_parser(PYTHON_LANGUAGE)
    tree = parser.parse(source_code.encode())
    root_node = tree.root_node

    imports = []
    for import_statement in root_node.children:
        if import_statement.type == 'import_statement':  # Example: `import math`
            imports.append(import_statement.text.decode('utf-8'))
        elif import_statement.type == 'import_from_statement':  # Example: `from os import path`
            imports.append(import_statement.text.decode('utf-8'))
    return imports


# Extraction logic for Java
def extract_java_imports(source_code):
    parser = get_parser(JAVA_LANGUAGE)
    tree = parser.parse(source_code.encode())
    root_node = tree.root_node

    imports = []
    for node in root_node.children:
        if node.type == 'import_declaration':  # Example: `import java.util.List`
            imports.append(source_code[node.start_byte:node.end_byte - 1])
    return imports


# Extraction logic for Kotlin
def extract_kotlin_imports(source_code):
    parser = get_parser(KOTLIN_LANGUAGE)
    tree = parser.parse(source_code.encode())
    root_node = tree.root_node

    imports = []
    for node in root_node.children:
        if node.type == 'import_list':  # Example: `import kotlin.collections.List`
            for import_declaration in node.children:
                imports.append(source_code[import_declaration.start_byte:import_declaration.end_byte])
    return imports


# Universal extraction based on file type
def extract_imports(source_code: str, file_extension: str):
    if file_extension == 'py':
        return extract_python_imports(source_code)
    elif file_extension == 'java':
        return extract_java_imports(source_code)
    elif file_extension == 'kt':
        return extract_kotlin_imports(source_code)
    else:
        # TODO: raise exception?
        return []
