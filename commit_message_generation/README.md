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
  * Models from [OpenAI API](https://platform.openai.com/docs/overview): implemented as [`OpenAIBackbone`](src/backbones/openai_backbone.py) class
  * Models from 🤗 [HuggingFace Hub](https://huggingface.co/): implemented as [`HuggingFaceBackbone`](src/backbones/hf_backbone.py) class
  * Models from [Together API](https://www.together.ai/): implemented as [`TogetherBackbone`](src/backbones/together_backbone.py) class
  * Models from [DeepSeek API](https://www.together.ai/): implemented as [`DeepSeekBackbone`](src/backbones/deepseek_backbone.py) class
* **Preprocessors:**
  * Simple preprocessing: implemented as [`SimpleCMGPreprocessor`](src/preprocessors/simple_diff_preprocessor.py) class
  * Simple preprocessing + truncation: implemented as [`TruncationCMGPreprocessor`](src/preprocessors/truncation_diff_preprocessor.py) class
  * BM25 retrieval: implemented as [`RetrievalCMGPreprocessor`](src/preprocessors/retrieval_preprocessor.py) class
  * Full modified files contents instead of diffs: implemented as [`FullFilesCMGPreprocessor`](src/preprocessors/full_files_preprocessor.py) class
  * Utility preprocessor that loads prebuilt contexts from a HF dataset: implemented as [`LoadFromDatasetPreprocessor`](src/preprocessors/load_from_dataset_preprocessor.py) class

* **Prompts:** 
  * Plain zero-shot prompt: implemented as [`SimpleCMGPrompt`](src/prompts/prompts.py) class
  * Detailed zero-shot prompt: implemented as [`DetailedCMGPrompt`](src/prompts/prompts.py) class
  * Detailed zero-shot prompt for Diff + BM25 setting: implemented as [`DetailedCMGPromptWContext`](src/prompts/prompts.py) class
  * Detailed zero-shot prompt for Full File setting: implemented as [`DetailedCMGPromptForFullFiles`](src/prompts/prompts.py) class
</details>

We also provide several `.yaml` configs as examples (see [Available Examples](#available-examples) section).
If you choose to use `.yaml` config, make sure to update `config_path` and `config_name` arguments accordingly: 
either in `hydra.main` decorator in [`run_baseline.py`](run_baseline.py) or by passing `--config-path` and `--config-name` command-line arguments.

## 🚀 Run

The main running script is [`run_baseline.py`](run_baseline.py).

* If you use Poetry, run: `poetry run python run_baseline.py`
* Otherwise, run: `python run_baseline.py`

In both cases, you can also add command-line arguments using [Hydra's override feature](https://hydra.cc/docs/advanced/override_grammar/basic/) (see [Available Examples](#available-examples) section for examples).

> There is also an asynchronous version [`arun_baseline.py`](arun_baseline.py). The main difference is that it always expects a YAML config to be stored under [`configs/async`](configs/async) and doesn't use override from Hydra.
> 
> * If you use Poetry, run: `poetry run python arun_baseline.py --config-name {your-config-basename}`
> * Otherwise, run: `python arun_baseline.py --config-name {your-config-basename}`

Additionally, there is a script that launches only preprocessor and saves preprocessed contexts for each commit.
It could be used in cases when preprocessing is sophisticated enough to do it on the fly (e.g., with retrieval).

* If you use Poetry, run: `poetry run python run_preprocessor.py`
* Otherwise, run: `python run_preprocessor.py`

In both cases, you can also add command-line arguments using [Hydra's override feature](https://hydra.cc/docs/advanced/override_grammar/basic/) (see [Available Examples](#available-examples) section for examples).

# Available examples

Together with the dataset, we release the results for several models.
They were obtained using this repository, and we provide `.yaml` configs examples under [`configs/examples`](configs/examples) folder. See also [`configs/async`](configs/async) for models that were launched with [`arun_baseline.py`](arun_baseline.py) and [`configs/retrieval`](configs/retrieval) for runs that saved preprocessed contexts via [`run_preprocessor.py`](run_preprocessor.py).
