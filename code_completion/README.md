# 🏟️ Long Code Arena Baselines
## Project Level Code Completion

This folder contains code for running baselines for Project Level Code Completion task in Long Code Arena benchmark.

We provide the implementation for the following baseline: a language model that is fed with differently composed context from a repository snapshot.

# How-to

## 💾 Install dependencies

* Change the working directory: `cd code_completion`
* To install all the dependencies run `poetry install`
* If you are going to use Flash Attention, run `poetry run pip install flash-attn --no-build-isolation`.
  * Refer to [official documentation](https://github.com/Dao-AILab/flash-attention?tab=readme-ov-file#installation-and-features) for more details.
* Some of the composers use [Tree Sitter](https://tree-sitter.github.io/tree-sitter/). Run `git clone https://github.com/tree-sitter/tree-sitter-python` to clone the Tree-sitter for Python.
  * Library will be compiled automatically in [`tree_sitter_parser/parser.py`](tree_sitter_parser/parser.py).
  * Refer to [official documentation](https://github.com/tree-sitter/py-tree-sitter?tab=readme-ov-file#setup) for more details.

## ⚙️ Configure a baseline

We use [Hydra](https://hydra.cc/docs/intro/) for configuration. Main config used for running experiments is located in [`eval/config/config.yaml`](eval/config/config.yaml).

### Add your model
* Base class to build the model is [`ModelBuilderBase`](model_hub/model_classes.py).
* To evaluate your model you need to add it to [registry](model_hub/model_registry.py).

### Add your context composer
* Base class for a composer is [`OneCompletonFileComposer`](eval/composers.py).
* To evaluate your composer you need to add it to `COMPOSERS` dictionary that is located in [eval/composers.py](eval/composers.py).

### Suported datasets:
* [`JetBrains-Research/lca-codegen-small`](https://huggingface.co/datasets/JetBrains-Research/lca-codegen-small)
* [`JetBrains-Research/lca-codegen-medium`](https://huggingface.co/datasets/JetBrains-Research/lca-codegen-medium)

## 🚀 Run

The main running script is [`eval/eval_pipeline.py`](eval/eval_pipeline.py).

* To start evaluation with Poetry, run: `poetry run python -m eval.eval_pipeline wandb_project_name=%project_name_1% wandb_project_name_generation=%project_name_2%`
  * `%project_name_1%` is a name for [wandb](https://wandb.ai/) project with the results of the next token prediction task to compare composers. Target metric is perplexity on completion file.
  * `%project_name_2%` is a name for [wandb](https://wandb.ai/) project with the results ot the one line code completion task with best

You can also add command-line arguments using [Hydra's override feature](https://hydra.cc/docs/advanced/override_grammar/basic/).

### Examples
* [Starcoder Base 7B](https://huggingface.co/bigcode/starcoderbase-7b) on small dataset
  * Command:
    ```
    poetry run python -m eval.eval_pipeline wandb_project_name=%project_name_1% wandb_project_name_generation=%project_name_2% dataset='JetBrains-Research/lca-codegen-small'
    +params=starcoderbase7b
    ```
* [CodeLlama 7B](https://huggingface.co/codellama/CodeLlama-7b-hf) in 4bit quantization with context window 8K on medium dataset  
  * Command:
    ```
    poetry run python -m eval.eval_pipeline wandb_project_name=%project_name_1% wandb_project_name_generation=%project_name_2% dataset='JetBrains-Research/lca-codegen-medium'
    +params=codellama7b_4bit params.inference_params.seq_max_len=8000
    ```
