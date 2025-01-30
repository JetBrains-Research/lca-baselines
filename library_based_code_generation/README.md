# BenchName Baselines
## Library-based code generation

This directory contains the code for running baselines for Library-based code generation task in BenchName benchmark.

We provide the implementation of baselines running inference via [OpenAI](https://platform.openai.com/docs/overview) and [Together.AI](https://www.together.ai/).
We evaluate multiple models in two settings: 
* Generating a program based on a plain instruction, without any repository-level information;
* Generating a program based on the instruction and top-20 method and class names from the library according to BM-25 with instruction as a reference.

# How-to

## 💾 Install dependencies

We provide dependencies for two Python dependencies managers: [pip](https://pip.pypa.io/en/stable/) and [Poetry](https://python-poetry.org/docs/). Poetry is preferred, `requirements.txt` is obtained by running `poetry export --with dev,eda --output requirements.txt`.

* If you prefer pip, run `pip install -r requirements.txt`
* If you prefer Poetry, run `poetry install`

## 🚀 Run

In order to evaluate models, run the [evaluation script](src/evaluation/evaluate.py).

* If you use Poetry, run: `poetry run python -m src.evaluation.evaluate`
* Otherwise, run: `python -m src.evaluation.evaluate`

The script will generate code with the available models and compute the metrics (`ChrF` and `API Recall`). 
If the code for the specific model is already available in the `results` directory, the evaluation script will just compute the metrics.


## ⚙️ Customize

* To implement more metrics, refer to the available [examples](src/metrics). Currently, we run `ChrF` and `API Recall`.
* To add more models, implement new instances of [abstract model](src/models/example_generation_model.py). We provide connectors to all models available via OpenAI and Together.AI.
* To work with code, we provide [examples of parsing](src/context) with the [tree-sitter](https://github.com/tree-sitter/py-tree-sitter) library.
