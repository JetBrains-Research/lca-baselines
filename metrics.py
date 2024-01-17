import os
import re
import json
import argparse
import numpy as np
import evaluate
from openai import OpenAI
from tqdm.auto import tqdm
from evaluate import load
from datasets import load_dataset
from utils.files_utils import load_config
from utils.openai_generation import GPT_generation


def extract_answer(string):
    pattern = r"\{.*?\}"
    # Finding all matches
    matches = re.findall(pattern, string.replace('\n', ' '))
    try:
        string2 = matches[0]
        ans = json.loads(string2)
        return int(ans['human_doc_idx'])
    except:
        try:
            string2 = matches[0]
            integer_pattern = r'\d+'
            integers = re.findall(integer_pattern, string2)
            return int(integers[0])
        except:
            integer_pattern = r'\d+'
            integers = re.findall(integer_pattern, string)
            return int(integers[-1])


def main(config, client):
    hf_api_key = config.get("hf_api_key")
    path_to_answers = config.get("save_dir", "./predictions")
    openai_key = config.get("openai_api_key")
    model_name = "gpt-3.5-turbo-16k"

    dataset = load_dataset("JetBrains-Research/lca-module-to-text",
                           token=hf_api_key)['train']

    predictions = []
    gold = []
    for path in os.listdir(path_to_answers):
        idx = int(path.split('.txt')[0])

        with open(f"{path_to_answers}/{path}", 'r') as f:
            pred = f.read()
        gld = dataset[idx]['target_text']

        predictions.append(pred)
        gold.append([gld])

    chrf = evaluate.load("chrf")
    chrf_results = chrf.compute(predictions=predictions, references=gold)

    rouge = evaluate.load("rouge")
    rouge_results = rouge.compute(predictions=predictions, references=gold)

    bertscore = load("bertscore")
    bertscore_results = bertscore.compute(predictions=predictions, references=gold, lang='en')

    counter_skip = 0
    metrics = []
    for pred, gld in tqdm(zip(predictions, gold), total=len(predictions), position=0, leave=True):
        try:
            prompt = "I have 2 text about the same code:\n\n\n"
            prompt += f"Documentation 1:\n{pred}"
            prompt += f"\n\n\nDocumentation 2:\n{gld[0]}"
            prompt += "\n\n\nDefine wich text is written by human. Return your answer in JSON format: {'human_doc_idx:': id (1 or 2)}. Return only JSON"
            answer = GPT_generation(client, prompt, model_name)
            answer_int = extract_answer(answer)

            prompt = "I have 2 text about the same code:\n\n\n"
            prompt += f"Documentation 1:\n{gld[0]}"
            prompt += f"\n\n\nDocumentation 2:\n{pred}"
            prompt += "\n\n\nDefine wich text is written by human. Return your answer in JSON format: {'human_doc_idx:': id (1 or 2)}. Return only JSON"
            answer2 = GPT_generation(client, prompt, model_name)
            answer_int2 = extract_answer(answer2)

            if answer_int == 1 and answer_int2 == 2:
                metrics.append(1)
            elif answer_int == 2 and answer_int2 == 1:
                metrics.append(0)
            else:
                metrics.append(0.5)
        except Exception as e:
            counter_skip += 1

    final_json = {
        'chrf': chrf_results['score'],
        'rouge1': rouge_results['rouge1'],
        'rouge2': rouge_results['rouge2'],
        'rougeL': rouge_results['rougeL'],
        "bertscore_f1": np.mean(bertscore_results['f1']),
        "bertscore_precision": np.mean(bertscore_results['precision']),
        "bertscore_recall": np.mean(bertscore_results['recall']),
        "chatgpt_mean": np.mean(metrics),
        "chatgpt_median": np.median(metrics),
        "chatgpt_counter_skip": counter_skip
    }
    with open(f'{path_to_answers}_answers.json', 'w') as f:
        json.dump(final_json, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Script with YAML config and command line arguments.")
    # Argument for YAML config file path
    parser.add_argument('--config', type=str, default="config.yaml", help='Path to the YAML configuration file')
    args = parser.parse_args()

    config = load_config(args.config)

    openai_key = config.get("openai_api_key")
    client = OpenAI(api_key=openai_key)

    main(config, client)
