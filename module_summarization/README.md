# 🏟️ Long Code Arena Baselines
## Module summarization

This directory contains the code for running baselines for the Module summarization task in the Long Code Arena benchmark.

We provide the implementation of baselines running inference via [OpenAI](https://platform.openai.com/docs/overview) and [Together.AI](https://www.together.ai/).
We generate documentation based on an intent (one sentence description of documentation content), target documentation name, and relevant code context. 

# How-to

## 💾 Install dependencies

We provide dependencies via the [Poetry](https://python-poetry.org/docs/) manager. 

* To install dependecies, run `poetry install`

## 🚀 Run 

#### Generation

In order to generate your predictions, add your parameters in the [configs](configs) directory and run: 

* `poetry run python chatgpt.py --config="configs/config_openai.yaml"` if you use [OpenAI](https://platform.openai.com/docs/overview) models;
* `poetry run python togetherai.py --config="configs/config_together.yaml"` if you use [Together.AI](https://www.together.ai/) models.

The script will generate predictions and put them into the `save_dir` directory from config.

#### Metrics 

To compare predicted and ground truth texts, we introduce the new metric based on LLM as an assessor. Our approach involves feeding the LLM with relevant code and two versions of documentation: the ground truth and the model-generated text. To mitigate variance and potential ordering effects in model responses, we calculate the probability that the generated documentation is superior by averaging the results of two queries:

```math
CompScore = \frac{ P(pred | LLM(code, pred, gold)) + P(pred | LLM(code, gold, pred))}{2}
```

In order to evaluate predictions, add your parameters in the [config](configs/config_eval.yaml) and run:
* `poetry run python metrics.py --config="configs/config_eval.yaml"`

The script will evaluate the predictions and save the results into the `results.json` file.
