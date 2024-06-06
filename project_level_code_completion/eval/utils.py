import json
import os

import numpy as np
import torch
from tqdm.auto import tqdm


def load_model_results(args):
    for repo_dir in tqdm(os.listdir(args.dataset_dir)):
        if not os.path.isdir(os.path.join(args.dataset_dir, repo_dir)):
            continue
        logits_path = os.path.join(args.dataset_dir, repo_dir, 'completion_logits.npy')
        logits_np = np.load(logits_path)
        # logits_pt[repo_dir] = torch.tensor(logits_np)
        tokens_path = os.path.join(args.dataset_dir, repo_dir, 'completion_tokens.npy')
        tokens_np = np.load(tokens_path)
        # tokens_pt[repo_dir] = torch.tensor(tokens_np)
        # ctxt_len_dict[repo_dir] = 0
        repo_id = int(repo_dir.split('_')[-1])
        yield repo_id, torch.tensor(logits_np), torch.tensor(tokens_np), 0


def save_results(eval_result, args):
    perplexity_dict = dict()
    bpc_dict = dict()
    for repo_id, next_token_prob in eval_result.items():
        perplexity = torch.exp(next_token_prob.mean()).item()
        perplexity_dict[repo_id] = perplexity
    mean_ppl = np.mean(list(perplexity_dict.values()))
    results_path = os.path.join(args.out_dir, 'ppl.json')
    with open(results_path, 'w') as json_file:
        json.dump([{'mean_ppl': mean_ppl}, perplexity_dict, ], json_file, indent=4)

    return mean_ppl
