# 🏟️ Long Code Arena Baselines
## Project-level code completion

This directory contains the code for running baselines for the Project-level code completion task in the [Long Code Arena benchmark](https://huggingface.co/spaces/JetBrains-Research/long-code-arena).

We provide the implementation for the following baseline: a language model that is fed with differently composed context from a repository snapshot.

The evaluation steps are the following:
* Choose the context composer:
    * Run the next token prediction for different context composers;
    * Choose the best one based on the lowest perplexity on the completion file.
* Evaluate the code completion with the composer:
    * Run one-line code completion with zero project context;
    * Run one-line code completion with a project context composed by the chosen context composer.

# How-to

## 💾 Install dependencies

* Change the working directory: `cd project_level_code_completion`
* install all the dependencies: `poetry install`
* If you are going to use Flash Attention, run `poetry run pip install flash-attn --no-build-isolation`.
  * Refer to the [official documentation](https://github.com/Dao-AILab/flash-attention?tab=readme-ov-file#installation-and-features) for more details.
  * If you are facing `ImportError` read more here: [Issue](https://github.com/Dao-AILab/flash-attention/issues/981)
* Some of the composers use [Tree Sitter](https://tree-sitter.github.io/tree-sitter/). Run `git clone https://github.com/tree-sitter/tree-sitter-python` to clone the Tree-sitter for Python.
  * The library will be compiled automatically in [`tree_sitter_parser/parser.py`](tree_sitter_parser/parser.py).
  * Refer to the [official documentation](https://github.com/tree-sitter/py-tree-sitter?tab=readme-ov-file#setup) for more details.

## ⚙️ Configure a baseline

We use [Hydra](https://hydra.cc/docs/intro/) for configuration. The main config used for running experiments is located in [`eval/config/config.yaml`](eval/config/config.yaml).

### Add your model
* Base class to build the model is [`ModelBuilderBase`](model_hub/model_classes.py).
* To evaluate your model, you need to add it to [registry](model_hub/model_registry.py).

### Add your context composer
* Base class for a composer is [`OneCompletionFileComposer`](composers/one_completion_file_composer.py).
* To evaluate your composer, you need to add it to `COMPOSERS` dictionary that is located in [composers/composer_registry.py](composers/composer_registry.py).

### Suported datasets:
* All configurations of [`JetBrains-Research/lca-project-level-code-completion`](https://huggingface.co/datasets/JetBrains-Research/lca-project-level-code-completion):
   * `small_context`
   * `medium_context`
   * `large_context`
   * `huge_context`

## 🚀 Run

The main running script is [`eval/eval_pipeline.py`](eval/eval_pipeline.py).

* To start evaluation with Poetry, run: `poetry run python -m eval.eval_pipeline params=codellama7b`.
You can also add command-line arguments using [Hydra's override feature](https://hydra.cc/docs/advanced/override_grammar/basic/).

### Hydra Config Main Parameters
* `params` – to choose a model to evaluate, possible values are filenames from [the directory](eval/config/params);
* `dataset` – to choose a dataset, possible values are filenames from [the directory](eval/config/dataset);
* `artifacts_dir` – where to put all the artifacts of evaluation:
    * the results are stored in `os.path.join(config.artifacts_dir, config.language, model_name, dataset_name)`;
* `wandb_project_name` – WandB project name for the step of choosing the composer;
* `wandb_project_name_generation` – WandB project name for the step of line generation.

### Examples
* [Starcoder Base 7B](https://huggingface.co/bigcode/starcoderbase-7b) on the `small_context` set:
  * Command:
    ```
    poetry run python -m eval.eval_pipeline dataset=small params=starcoderbase7b
    ```
* [CodeLlama 7B](https://huggingface.co/codellama/CodeLlama-7b-hf) in 4bit quantization with context window 8K on the `medium_context` dataset  
  * Command:
    ```
    poetry run python -m eval.eval_pipeline dataset=medium params=codellama7b_4bit params.inference_params.seq_max_len=8000
    ```
