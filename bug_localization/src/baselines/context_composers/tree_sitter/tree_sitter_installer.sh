rm -rf ./tree-sitter-java
git clone https://github.com/tree-sitter/tree-sitter-java.git
cd ./tree-sitter-java
make

rm -rf ./tree-sitter-kotlin
git clone https://github.com/tree-sitter/tree-sitter-kotlin.git
cd ./tree-sitter-kotlin
make

rm -rf ./tree-sitter-python
git clone https://github.com/tree-sitter/tree-sitter-python.git
cd ./tree-sitter-python
make
