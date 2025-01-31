import os
import argparse
import torch 
import json
import numpy as np
from tqdm.auto import tqdm
from datasets import load_dataset
from utils.context_utils import collect_good_context, trim_context
from utils.files_utils import load_config
from transformers import AutoModelForCausalLM, AutoTokenizer
from utils.scorer import OptionsScoringModel


def get_metric(scorer, intent, code_context, gold_doc, pred_doc):
    
    prompt = f'I have 2 different documentations about {intent}. Decide which documentation is better: documentation A or documentation B.\n\n' 
    prompt += f'My code:\n\n{code_context}\n\n\n\n'
    prompt += f'Documentation A:\n\n{gold_doc}\n\n\n\n'
    prompt += f'Documentation B:\n\n{pred_doc}\n\n\n\n'
    prompt += 'Better documentation is documentation '
    
    options = ["A", "B"]
    unnorm_logprobs = scorer.score_options(prompt, options)
    norm_probs1 = torch.exp(torch.log_softmax(unnorm_logprobs, dim=0))
    
    prompt = f'I have 2 different documentations about {intent}. Decide which documentation is better: documentation A or documentation B.\n\n' 
    prompt += f'My code:\n\n{code_context}\n\n\n\n'
    prompt += f'Documentation A:\n\n{pred_doc}\n\n\n\n'
    prompt += f'Documentation B:\n\n{gold_doc}\n\n\n\n'
    prompt += 'Better documentation is documentation '
    unnorm_logprobs = scorer.score_options(prompt, options)
    norm_probs2 = torch.exp(torch.log_softmax(unnorm_logprobs, dim=0))
    
    p_better1 = (norm_probs1[1] + norm_probs2[0]) / 2 
    return float(p_better1)


def score_one_model(scorer, dataset, direct, max_cont_len, tokenizer, use_pbar=False):
    golds, preds, intents, codes = [], [], [], []
    
    for idx in range(len(dataset)):
        with open(f"{direct}/{idx}.txt", 'r') as f:
            pred = f.read()
        gld = dataset[idx]['target_text']
        golds.append(gld)
        preds.append(pred)
            
        intents.append(dataset[idx]['intent'])
        codes.append(trim_context(dataset[idx]['relevant_code_context'], tokenizer, max_cont_len))
    
    pbar = range(len(golds))
    if use_pbar:
        pbar = tqdm(pbar, total=len(pbar), position=0, leave=True)
        
    metrics = []
    for idx in pbar:
        m = get_metric(scorer, intents[idx], codes[idx], golds[idx], preds[idx])
        metrics.append(m)
    return metrics


def score_gold(scorer, dataset, max_cont_len, tokenizer, use_pbar=False):
    golds, preds, intents, codes = [], [], [], []
    
    for idx in range(len(dataset)):
        gld = dataset[idx]['target_text']
        golds.append(gld)
        preds.append(gld)

        intents.append(dataset[idx]['intent'])
        codes.append(trim_context(dataset[idx]['relevant_code_context'], tokenizer, max_cont_len))
    
    pbar = range(len(golds))
    if use_pbar:
        pbar = tqdm(pbar, total=len(pbar), position=0, leave=True)
        
    metrics = []
    for idx in pbar:
        m = get_metric(scorer, intents[idx], codes[idx], golds[idx], preds[idx])
        metrics.append(m)
    return metrics


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Script with YAML config and command line arguments."
    )
    # Argument for YAML config file path
    parser.add_argument('--config', type=str, 
                        default="configs/config_eval.yaml", 
                        help='Path to the YAML configuration file')
    args = parser.parse_args()
    
    config = load_config(args.config)
    
    hf_api_key = config.get("hf_api_key")
    api_key = config.get("api_key")
    model_name = config.get("model_name")
    device = config.get("device")
    hf_tokenizer_checkpoint = config.get("hf_tokenizer_checkpoint")
    max_context_toks = config.get("max_context_toks", None)
    
    tokenizer = AutoTokenizer.from_pretrained(hf_tokenizer_checkpoint, 
                                              token=hf_api_key)
    

    scorer = OptionsScoringModel(model_name, device)
    dataset = load_dataset("icmlbenchname/module-summarization",
                               token=hf_api_key)['test']
    
    path_to_configs = 'configs'
    
    path2metric = {}
    for root, dirs, files in os.walk(path_to_configs):
        for file in tqdm(files[:]):
            file_path = os.path.join(root, file)
            if '_eval' not in file_path:
                config = load_config(file_path)
                save_dir = config.get("save_dir")
                
                model_metric = np.mean(
                    score_one_model(
                        scorer, dataset, save_dir, 
                        max_context_toks, tokenizer, True
                    )
                )
                
                print(f'Metric for {save_dir} = {model_metric}')
                path2metric[save_dir] = model_metric

    with open('result_gold.json', 'w') as f:
        json.dump(path2metric, f)