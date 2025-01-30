# BenchName Baselines
## Bug localization

This directory contains the code for the Bug localization benchmark. The challenge is: 
given an issue with the bug description and the repository code in the state where the issue is reproducible, identify the files within the project that need to be modified to address the reported bug.

We provide scripts for [data collection and processing](./src/data), [data exploratory analysis](./src/notebooks), as well as several [baselines implementations](./src/baselines) for solving the task with [the calculation of evaluation metrics](./src/notebooks).

## 💾 Install dependencies
We provide dependencies for the pip dependency manager, so please run the following command to install all the required packages:
```shell
pip install -r requirements.txt
```

## 🤗 Dataset

All the data is stored in [HuggingFace 🤗](icmlbenchname/bug-localization). It contains:

* A **dataset** with the bug localization data (issue description, SHA of the repo in initial state, and SHA of the repo after fixing the issue).
You can access the data using the [datasets](https://huggingface.co/docs/datasets/en/index) library:
    ```python3
    from datasets import load_dataset
    
    # Select a configuration from ["py", "java", "kt"]
    configuration = "py"
    # Select a split from ["dev", "train", "test"]
    split = "dev"
    # Load data
    dataset = load_dataset("icmlbenchname/bug-localization", configuration, split=split)
    ```
    ...where the labels are:\
    `dev` — all the collected data;\
    `test` — manually selected data ([labeling artifacts](https://docs.google.com/spreadsheets/d/1cEyFHjse-iUYQlUO7GO5KpqkvJ3wu6vheou4W61TMOg/edit?usp=sharing));\
    `train` — all the collected data that is not in `test`;

    ...and configurations are:\
    `py` — only `.py` files in diff;\
    `java` — only `.java` files in diff;\
    `kt` — only `.kt` files in diff;


* **Archived repos** (from which we can extract repository content at different stages and get diffs that contains bugs fixes).\

## ⚙️ Baselines

* Embedding-based:
  * [TF-IDF](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html#sklearn.feature_extraction.text.TfidfVectorizer);
  * [GTE](https://huggingface.co/thenlper/gte-large);
  * [CodeT5](https://huggingface.co/Salesforce/codet5p-110m-embedding);
  * [BM25]().
  
* Chat-based:
  * [GPT3.5](https://platform.openai.com/docs/models/gpt-3-5-turbo);
  * [GPT4](https://platform.openai.com/docs/models/gpt-4).
