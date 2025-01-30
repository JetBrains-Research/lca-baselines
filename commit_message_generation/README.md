# BenchName Baselines
## Commit message generation

This directory contains the code for running baselines for the Commit message generation (CMG) task in the BenchName benchmark.

We provide the implementation for the following baseline: a language model that is fed with a zero-shot prompt with a simple instruction and a commit diff.

# How-to

## 💾 Install dependencies

We provide dependencies for two Python dependencies managers: [pip](https://pip.pypa.io/en/stable/) and [Poetry](https://python-poetry.org/docs/). Poetry is preferred, `requirements.txt` is obtained by running `poetry export --with dev,eda --output requirements.txt`.

* If you prefer pip, run `pip install -r requirements.txt`
* If you prefer Poetry, run `poetry install`

## ⚙️ Configure a baseline

We use [Hydra](https://hydra.cc/docs/intro/) for configuration. The main config used for running experiments is `BaselineConfig`, located in [`configs/baseline_config.py`](configs/baseline_config.py). 
Refer to Hydra documentation and to documentation of this class for more details.

### Supported configurations

We provide the implementation for the following baseline: a language model that is fed with a zero-shot prompt with a simple instruction and a commit diff.

This baseline consists of the following configurable components:
* **Models / Backbones:** stored under [`src/backbones`](src/backbones), base class is [`CMGBackbone`](src/backbones/base_backbone.py)
* **Preprocessors:** stored under [`src/preprocessors`](src/preprocessors), base class is [`CMGPreprocessor`](src/preprocessors/base_preprocessor.py)
* **Prompts:** stored under [`src/prompts`](src/prompts), base class is [`CMGPrompt`](src/prompts/base_prompt.py)

<details>
<summary>💛 Click here to view the currently supported options for each component.</summary>

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

**Note.** The configs and the commands are provided for a single seed value, which is controlled by `backbone.parameters.seed` for OpenAI models and `backbone.seed` for models from HuggingFace Hub. We averaged the results across three seeds. For convenience, you can use [Hydra's multi-run functionality](https://hydra.cc/docs/tutorials/basic/running_your_app/multi-run/) to launch three subsequent runs with different seeds. 

## OpenAI models

* GPT-3.5 Turbo (16k)
  * Config: [`gpt_3.5_16k.yaml`](configs/examples/gpt_3.5_16k.yaml)
  * Command:
    ```
    poetry run python run_baseline.py +data_src=hf data_src.hub_name=icmlbenchname/ci-builds-repair data_src.configs="[commitchronicle-py-long]" +preprocessor=simple preprocessor.include_path=true +backbone=openai +backbone/prompt=detailed backbone.model_name=gpt-3.5-turbo-16k ++backbone.parameters.temperature=0.8 ++backbone.parameters.seed=2687987020 logger.name=gpt_3.5_16k-detailed
    ```
* GPT-4
  * Config: [`gpt_4_8k.yaml`](configs/examples/gpt_4_8k.yaml)
  * Command:
    ```
    poetry run python run_baseline.py +data_src=hf data_src.hub_name=icmlbenchname/ci-builds-repair data_src.configs="[commitchronicle-py-long]" +preprocessor=truncation preprocessor.include_path=true preprocessor.max_num_tokens=8000 +backbone=openai +backbone/prompt=detailed backbone.model_name=gpt-4 ++backbone.parameters.temperature=0.8 ++backbone.parameters.seed=2687987020 logger.name=gpt_4-8k-detailed
    ```
* GPT-4 Turbo
  * Config: [`gpt-4-1106-preview.yaml`](configs/examples/gpt-4-1106-preview.yaml)
  * Command:
    ```
    poetry run python run_baseline.py +data_src=hf data_src.hub_name=icmlbenchname/ci-builds-repair data_src.configs="[commitchronicle-py-long]" +preprocessor=simple preprocessor.include_path=true +backbone=openai +backbone/prompt=detailed backbone.model_name=gpt-4-1106-preview ++backbone.parameters.temperature=0.8 ++backbone.parameters.seed=2687987020 logger.name=gpt-4-1106-preview-detailed
    ```
## 🤗 Models from HuggingFace Hub

> Note: Most of the models were launched with [FlashAttention2](https://github.com/Dao-AILab/flash-attention) enabled, which is controlled by an argument under `backbone.model_kwargs` config key. FlashAttention2 is not included in the requirements for this repository, please, install it separately, following [official guidelines](https://github.com/Dao-AILab/flash-attention?tab=readme-ov-file#installation-and-features).

* [CodeLlama-7b (instruct)](https://huggingface.co/codellama/CodeLlama-7b-Instruct-hf)
  * Config: [`codellama_7b.yaml`](configs/examples/codellama_7b.yaml)
  * Command:
    ```
    poetry run python run_baseline.py +data_src=hf data_src.hub_name=icmlbenchname/ci-builds-repair data_src.configs="[default]" +preprocessor=simple preprocessor.include_path=true +backbone=hf backbone.model_name=codellama/CodeLlama-7b-Instruct-hf backbone.is_encoder_decoder=false backbone.device=cuda ++backbone.model_kwargs.attn_implementation=flash_attention_2 +backbone/prompt=detailed backbone.generation.max_new_tokens=512 backbone.seed=2687987020 logger.name=codellama7b_detailed
    ```
* [CodeLlama-13b (instruct)](https://huggingface.co/codellama/CodeLlama-13b-Instruct-hf)
  * Config: [`codellama_13b.yaml`](configs/examples/codellama_13b.yaml)
  * Command:
    ```
    poetry run python run_baseline.py +data_src=hf data_src.hub_name=icmlbenchname/ci-builds-repair data_src.configs="[default]" +preprocessor=simple preprocessor.include_path=true +backbone=hf backbone.model_name=codellama/CodeLlama-13b-Instruct-hf backbone.is_encoder_decoder=false backbone.device=cuda ++backbone.model_kwargs.attn_implementation=flash_attention_2 +backbone/prompt=detailed backbone.generation.max_new_tokens=512 backbone.seed=2687987020 logger.name=codellama13b_detailed
    ```
* [CodeLlama-34b (instruct)](https://huggingface.co/codellama/CodeLlama-34b-Instruct-hf)
  * Config: [`codellama_34b.yaml`](configs/examples/codellama_34b.yaml)
  * Command:
    ```
    poetry run python run_baseline.py +data_src=hf data_src.hub_name=icmlbenchname/ci-builds-repair data_src.configs="[default]" +preprocessor=simple preprocessor.include_path=true +backbone=hf backbone.model_name=codellama/CodeLlama-34b-Instruct-hf backbone.is_encoder_decoder=false backbone.device=cuda ++backbone.model_kwargs.attn_implementation=flash_attention_2 +backbone/prompt=detailed backbone.generation.max_new_tokens=512 backbone.seed=2687987020 logger.name=codellama34b_detailed
    ```
* [DeepSeek Coder-1.3b (instruct)](https://huggingface.co/deepseek-ai/deepseek-coder-1.3b-instruct)
  * Config: [`deepseek-coder-1.3b.yaml`](configs/examples/deepseek-coder-1.3b.yaml)
  * Command:
    ```
    poetry run python run_baseline.py +data_src=hf data_src.hub_name=icmlbenchname/ci-builds-repair data_src.configs="[default]" +preprocessor=simple preprocessor.include_path=true +backbone=hf backbone.model_name=deepseek-ai/deepseek-coder-1.3b-instruct backbone.is_encoder_decoder=false backbone.device=cuda ++backbone.model_kwargs.attn_implementation=flash_attention_2 +backbone/prompt=detailed backbone.generation.max_new_tokens=512 backbone.seed=2687987020 logger.name=deepseek-coder-1.3b_detailed
    ```
* [DeepSeek Coder-6.7b (instruct)](https://huggingface.co/deepseek-ai/deepseek-coder-6.7b-instruct)
  * Config: [`deepseek-coder-6.7b.yaml`](configs/examples/deepseek-coder-6.7b.yaml)
  * Command:
    ```
    poetry run python run_baseline.py +data_src=hf data_src.hub_name=icmlbenchname/ci-builds-repair data_src.configs="[default]" +preprocessor=simple preprocessor.include_path=true +backbone=hf backbone.model_name=deepseek-ai/deepseek-coder-6.7b-instruct backbone.is_encoder_decoder=false backbone.device=cuda ++backbone.model_kwargs.attn_implementation=flash_attention_2 +backbone/prompt=detailed backbone.generation.max_new_tokens=512 backbone.seed=2687987020 logger.name=deepseek-coder-6.7b_detailed
    ```
* [DeepSeek Coder-33b (instruct)](https://huggingface.co/deepseek-ai/deepseek-coder-33b-instruct)
  * Config: [`deepseek-coder-33b.yaml`](configs/examples/deepseek-coder-33b.yaml)
  * Command:
    ```
    poetry run python run_baseline.py +data_src=hf data_src.hub_name=icmlbenchname/ci-builds-repair data_src.configs="[default]" +preprocessor=simple preprocessor.include_path=true +backbone=hf backbone.model_name=deepseek-ai/deepseek-coder-33b-instruct backbone.is_encoder_decoder=false backbone.device=cuda ++backbone.model_kwargs.attn_implementation=flash_attention_2 +backbone/prompt=detailed backbone.generation.max_new_tokens=512 backbone.seed=2687987020 logger.name=deepseek-coder-33b_detailed
    ```
* [Mistral-7b (instruct, v0.2)](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2)
  * Config: [`mistral_7b.yaml`](configs/examples/mistral_7b.yaml)
  * Command:
    ```
    poetry run python run_baseline.py +data_src=hf data_src.hub_name=icmlbenchname/ci-builds-repair data_src.configs="[default]" +preprocessor=simple preprocessor.include_path=true +backbone=hf backbone.model_name=mistralai/Mistral-7B-Instruct-v0.2 backbone.is_encoder_decoder=false backbone.device=cuda ++backbone.model_kwargs.attn_implementation=flash_attention_2 +backbone/prompt=detailed backbone.generation.max_new_tokens=512 backbone.seed=2687987020 logger.name=mistral7b_detailed
    ```
* [Mixtral (instruct, 8bit)](https://huggingface.co/mistralai/Mixtral-8x7B-Instruct-v0.1)
  * Config: [`mixtral_8bit.yaml`](configs/examples/mixtral_8bit.yaml)
  * Command:
    ```
    poetry run python run_baseline.py +data_src=hf data_src.hub_name=icmlbenchname/ci-builds-repair data_src.configs="[default]" +preprocessor=simple preprocessor.include_path=true +backbone=hf backbone.model_name=mistralai/Mixtral-8x7B-Instruct-v0.1 backbone.is_encoder_decoder=false backbone.device=cuda ++backbone.model_kwargs.load_in_8bit=true ++backbone.model_kwargs.attn_implementation=flash_attention_2 +backbone/prompt=detailed backbone.generation.max_new_tokens=512 backbone.seed=2687987020 logger.name=mixtral_8bit_detailed
    ```
