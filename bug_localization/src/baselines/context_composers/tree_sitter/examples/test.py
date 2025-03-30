from src.baselines.context_composers.tree_sitter.utils import extract_imports

if __name__ == '__main__':
    # Provide file paths for testing
    python_file = "./examples/example.py"
    java_file = "./examples/example.java"
    kotlin_file = "./examples/example.kt"

    for file in ["example.py", "example.java", "example.kt"]:
        with open(file, 'r') as f:
            source_code = f.read()
        extension = file.split(".")[-1]
        print(f"{extension.capitalize()} imports:", extract_imports(source_code, extension))
