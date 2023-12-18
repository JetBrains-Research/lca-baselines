# 🏟️ Long Code Arena Baselines
## Commit Message Generation

This folder contains code for running baselines for Commit Message Generation (CMG) task in Long Code Arena benchmark.

We provide the implementation for the following baseline: a language model that is fed with a zero-shot prompt with a simple instruction and a commit diff.

# How-to

## 💾 Install dependencies

We provide dependencies for two Python dependencies managers: [pip](https://pip.pypa.io/en/stable/) and [Poetry](https://python-poetry.org/docs/). Poetry is preferred, `requirements.txt` is obtained by running `poetry export --with dev,eda --output requirements.txt`.

* If you prefer pip, run `pip install -r requirements.txt`
* If you prefer Poetry, run `poetry install`

## ⚙️ Configure a baseline

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
  * Models from 🤗 HuggingFace Hub: implemented as [`HuggingFaceBackbone`](src/backbones/hf_backbone.py) class
* **Preprocessors:**
  * Simple preprocessing: implemented as [`SimpleCMGPreprocessor`](src/preprocessors/simple_diff_preprocessor.py) class
  * Simple preprocessing + truncation: implemented as [`TruncationCMGPreprocessor`](src/preprocessors/truncation_diff_preprocessor.py) class
* **Prompts:** 
  * Plain zero-shot prompt: implemented as [`SimpleCMGPrompt`](src/prompts/prompts.py) class
  * Detailed zero-shot prompt: implemented as [`DetailedCMGPrompt`](src/prompts/prompts.py) class
</details>

We also provide several `.yaml` configs as examples (see [Available Examples](#available-examples) section).
If you choose to use `.yaml` config, make sure to update `config_path` and `config_name` arguments accordingly: 
either in `hydra.main` decorator in [`run_baseline.py`](run_baseline.py) or by passing `--config-path` and `--config-name` command-line arguments.

## 🚀 Run

The main running script is [`run_baseline.py`](run_baseline.py).

* If you use Poetry, run: `poetry run python run_baseline.py`
* Otherwise, run: `python run_baseline.py`

In both cases, you can also add command-line arguments using [Hydra's override feature](https://hydra.cc/docs/advanced/override_grammar/basic/) (see [Available Examples](#available-examples) section for examples).

# Available examples

Together with the dataset, we release the results for several models.
They were obtained using this repository, 
and we provide the exact commands for each of them as well as `.yaml` configs examples under [`configs/examples`](configs/examples) folder.

## OpenAI models

* GPT-3.5 Turbo
  * Config: [`gpt_3.5_16k.yaml`](configs/examples/gpt_3.5_16k.yaml)
  * Command:
    ```
    poetry run python run_baseline.py +data_src=hf data_src.hub_name=JetBrains-Research/lca-cmg data_src.configs=[commitchronicle-py-long] +preprocessor=simple preprocessor.include_path=true +backbone=openai +backbone/prompt=detailed backbone.model_name=gpt-3.5-turbo-16k ++backbone.parameters.temperature=0.8 ++backbone.parameters.seed=2687987020
    ```
* GPT-4
  * Config: [`gpt_4.yaml`](configs/examples/gpt_4.yaml)
  * Command:
    ```
    poetry run python run_baseline.py +data_src=hf data_src.hub_name=JetBrains-Research/lca-cmg data_src.configs=[commitchronicle-py-long] +preprocessor=truncation preprocessor.include_path=true preprocessor.max_num_tokens=8000 +backbone=openai +backbone/prompt=detailed backbone.model_name=gpt-4 ++backbone.parameters.temperature=0.8 ++backbone.parameters.seed=2687987020
    ```
## 🤗 Models from HuggingFace Hub

* [CodeT5](https://huggingface.co/JetBrains-Research/cmg-codet5-without-history)
  * Config: [`cmg_codet5.yaml`](configs/examples/cmg_codet5.yaml)
  * Command:
    ```
    poetry run python run_baseline.py +data_src=hf data_src.hub_name=JetBrains-Research/lca-cmg data_src.configs=[commitchronicle-py-long] +preprocessor=simple preprocessor.include_path=true +backbone=hf backbone.model_name=JetBrains-Research/cmg-codet5-without-history backbone.is_encoder_decoder=true backbone.device=cuda backbone.seed=2687987020
    ```
* [CodeLlama-7b (instruct)](https://huggingface.co/codellama/CodeLlama-7b-Instruct-hf)
  * Config: [`codellama_7b.yaml`](configs/examples/codellama_7b.yaml)
  * Command:
    ```
    poetry run python run_baseline.py +data_src=hf data_src.hub_name=JetBrains-Research/lca-cmg data_src.configs=[commitchronicle-py-long] +preprocessor=simple preprocessor.include_path=true +backbone=hf backbone.model_name=codellama/CodeLlama-7b-Instruct-hf backbone.is_encoder_decoder=false backbone.device=cuda backbone.model_kwargs.load_in_4bit=true ++backbone.model_kwargs.attn_implementation=flash_attention_2 +backbone/prompt=detailed backbone.generation.max_new_tokens=512 backbone.seed=2687987020
    ```
  * Note: This model was launched with 4-bit quantization and with [FlashAttention2](https://github.com/Dao-AILab/flash-attention) enabled, which is controlled by arguments under `backbone.model_kwargs` key. FlashAttention2 is not included in the requirements for this repository, please, install it separately, following [official guidelines](https://github.com/Dao-AILab/flash-attention?tab=readme-ov-file#installation-and-features).
* [Mistral-7b (instruct, v0.2)](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2)
  * Config: [`mistral_7b.yaml`](configs/examples/mistral_7b.yaml)
  * Command:
    ```
    poetry run python run_baseline.py +data_src=hf data_src.hub_name=JetBrains-Research/lca-cmg data_src.configs=[commitchronicle-py-long] +preprocessor=simple preprocessor.include_path=true +backbone=hf backbone.model_name=mistralai/Mistral-7B-Instruct-v0.2 backbone.is_encoder_decoder=false backbone.device=cuda backbone.model_kwargs.load_in_4bit=true ++backbone.model_kwargs.attn_implementation=flash_attention_2 +backbone/prompt=detailed backbone.generation.max_new_tokens=512 backbone.seed=2687987020
    ```
  * Note: This model was launched with 4-bit quantization and with [FlashAttention2](https://github.com/Dao-AILab/flash-attention) enabled, which is controlled by arguments under `backbone.model_kwargs` key. FlashAttention2 is not included in the requirements for this repository, please, install it separately, following [official guidelines](https://github.com/Dao-AILab/flash-attention?tab=readme-ov-file#installation-and-features).
