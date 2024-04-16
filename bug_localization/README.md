# Bug Localization

This folder contains code for **Bug Localization** benchmark. Challenge: 
given an issue with bug description, identify the files within the project that need to be modified
to address the reported bug.

We provide scripts for [data collection and processing](./src/data), [data exploratory analysis](./src/notebooks) as well as several [baselines implementations](./src/baselines) for the task solution.
## 💾 Install dependencies
We provide dependencies for pip dependency manager, so please run the following command to install all required packages:
```shell
pip install -r requirements.txt
```
Bug Localization task: given an issue with bug description, identify the files within the project that need to be modified to address the reported bug

## 🤗 Load data
All data is stored in [HuggingFace 🤗](https://huggingface.co/datasets/tiginamaria/bug-localization). It contains:

* Dataset with bug localization data (with issue description, sha of repo with initial state and to the state after issue fixation).
You can access data using [datasets](https://huggingface.co/docs/datasets/en/index) library:
    ```python3
    from datasets import load_dataset
    
    # Select a configuration from ["py", "java", "kt", "mixed"]
    configuration = "py"
    # Select a split from ["dev", "train", "test"]
    split = "dev"
    # Load data
    dataset = load_dataset("JetBrains-Research/lca-bug-localization", configuration, split=split)
    ```
    where labels are:\
    `dev` - all collected data\
    `test` - manually selected data ([labeling artifacts](https://docs.google.com/spreadsheets/d/1cEyFHjse-iUYQlUO7GO5KpqkvJ3wu6vheou4W61TMOg/edit?usp=sharing))\
    `train` - all collected data which is not in test\
    and configurations are:\
    `py` -- only `.py` files in diff\
    `java` -- only `.java` files in diff\
    `kt` -- only `.kt` files in diff\
    `mixed` -- at least on of the `.py`, `.java` or `.kt` file and maybe file(s) with another extensions in diff


* Archived repos (from which we can extract repo content on different stages and get diffs which contains bugs fixations).\
They are stored in `.tar.gz` so you need to run script to load them and unzip:
  1. Set `repos_path` in [config](configs/data/hf_data.yaml) to directory where you want to store repos
  2. Run [load_data_from_hf.py](./src/load_data_from_hf.py) which will load all repos from HF and unzip them

## ⚙️ Run Baseline

* Embedding-based
  * [TF-IDF](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html#sklearn.feature_extraction.text.TfidfVectorizer)
  * [GTE](https://huggingface.co/thenlper/gte-large)
  * [CodeT5](https://huggingface.co/Salesforce/codet5p-110m-embedding)
  * [BM25](https://platform.openai.com/docs/models/gpt-3-5-turbo)
  
* Name-based
  * [GPT3.5](https://platform.openai.com/docs/models/gpt-3-5-turbo)
  * [GPT4](https://platform.openai.com/docs/models/gpt-3-5-turbo)
  * [Cloud 2](https://platform.openai.com/docs/models/gpt-3-5-turbo)
  * [CodeLLama](https://platform.openai.com/docs/models/gpt-3-5-turbo)
  * [Mistral](https://platform.openai.com/docs/models/gpt-3-5-turbo)
