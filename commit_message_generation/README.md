# 🏟️ Long Code Arena Baselines
## Commit Message Generation

This folder contains code for running baselines for Commit Message Generation (CMG) task in Long Code Arena benchmark.

We provide the implementation for the following baseline: a language model that is fed with a zero-shot prompt with a simple instruction and a commit diff.

> Our dataset for Commit Message Generation task is available on :hugs: HuggingFace: [link](https://huggingface.co/datasets/JetBrains-Research/lca-cmg)

# How-to

## 💾 Install dependencies

We provide dependencies for two Python dependencies managers: [pip](https://pip.pypa.io/en/stable/) and [Poetry](https://python-poetry.org/docs/). Poetry is preferred, `requirements.txt` is obtained by running `poetry export --with dev,eda --output requirements.txt`.

* If you prefer pip, run `pip install -r requirements.txt`
* If you prefer Poetry, run `poetry install`

## ⚙️ Configure a baseline

> 🚧 TODO: expand description

We use [Hydra](https://hydra.cc/docs/intro/) for configuration. Main config used for running experiments is `BaselineConfig`, located in [`configs/baseline_config.py`](configs/baseline_config.py). 
Refer to Hydra documentation and to documentation of this class for more details.

### Supported configurations

We provide the implementation for the following baseline: a language model that is fed with a zero-shot prompt with a simple instruction and a commit diff.

This baseline consists of the following configurable components:
* **Models / Backbones:** stored under [`src/backbones`](src/backbones), base class is [`CMGBackbone`](src/backbones/base_backbone.py)
* **Preprocessors:** stored under [`src/preprocessors`](src/preprocessors), base class is [`CMGPreprocessor`](src/preprocessors/base_preprocessor.py)
* **Prompts:** stored under [`src/prompts`](src/prompts), base class is [`CMGPrompt`](src/prompts/base_prompt.py)

<details>
<summary>💛 Click here to view currently supported options for each component.</summary>

* **Models / Backbones:**
  * Models from OpenAI API: implemented as [`OpenAIBackbone`](src/backbones/openai_backbone.py) class
  * Models from :hugs: HuggingFace Hub: implemented as [`HuggingFaceBackbone`](src/backbones/hf_backbone.py) class
* **Preprocessors:**
  * Simple preprocessing: implemented as [`SimpleCMGPreprocessor`](src/preprocessors/simple_diff_preprocessor.py) class
  * Simple preprocessing + truncation: implemented as [`TruncationCMGPreprocessor`](src/preprocessors/truncation_diff_preprocessor.py) class
* **Prompts:** 
  * Plain zero-shot prompt: implemented as [`SimpleCMGPrompt`](src/prompts/prompts.py) class
  * Detailed zero-shot prompt: implemented as [`DetailedCMGPrompt`](src/prompts/prompts.py) class
</details>

### Available examples

> 🚧 TODO: actually add these examples

Also, we provide several examples of `.yaml` configs under [`configs/examples`](configs/examples) folder.

If you choose to use one of these, make sure to 
update `config_path` and `config_name` arguments accordingly: either in `hydra.main` decorator in [`run_baseline.py`](run_baseline.py) or by passing `--config-path` and `--config-name` command-line arguments.

## 🚀 Run

The main running script is [`run_baseline.py`](run_baseline.py).

* If you use Poetry, run: `poetry run python run_baseline.py`
* Otherwise, run: `python run_baseline.py`

In both cases, you can also add command-line arguments using [Hydra's override feature](https://hydra.cc/docs/advanced/override_grammar/basic/). 
For instance, here is the command we used to launch Mistral-7b model:

```
poetry run python run_baseline.py +data_src=hf data_src.hub_name=JetBrains-Research/lca-cmg data_src.configs=commitchronicle-py-long +preprocessor=simple preprocessor.include_path=true logger.name=commitchronicle_py_long_mistral7b +backbone=hf backbone.model_name=mistralai/Mistral-7B-Instruct-v0.1 backbone.is_encoder_decoder=false backbone.device=cuda backbone.model_kwargs.load_in_4bit=true +backbone/prompt=detailed backbone.generation.max_new_tokens=256
```
