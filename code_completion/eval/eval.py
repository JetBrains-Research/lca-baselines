import argparse
import os

import torch
import torch.nn.functional as F

from eval.utils import save_results, load_model_results


@torch.no_grad()
def evaluate(args):
    next_token_prob_dict = dict()

    data_iterator = load_model_results(args)

    for repo_id, logits, tokens, ctx_len in data_iterator:
        logits = logits[..., ctx_len:-1, :].contiguous().to(args.device)
        labels = tokens[..., (ctx_len + 1):].contiguous().to(args.device)

        if (logits >= 0).all() and (torch.abs(logits.sum(dim=-1) - 1.) < 1e4).all():
            probs = logits
        else:
            probs = F.softmax(logits.float(), dim=-1)

        next_token_prob = torch.index_select(probs, dim=-1, index=labels)

        next_token_prob = next_token_prob.diagonal()
        next_token_prob = - torch.log(next_token_prob)

        # Another implementation:
        # loss_fn = nn.CrossEntropyLoss(reduction='none')
        # next_token_prob = loss_fn(logits, labels)

        next_token_prob_dict[int(repo_id)] = next_token_prob

    mean_ppl = save_results(next_token_prob_dict, args)
    return mean_ppl


if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--device', type=str, default='cpu', help='What device to use for evaluation.')
    argparser.add_argument('--out_dir', type=str, default='lca/code_generation/out_dir', help='Directory to save results of the evaluation.')
    argparser.add_argument('--dataset_dir', type=str, default='lca/code_generation/data', help='Directory with saved logits.')

    args = argparser.parse_args()

    if not os.path.exists(args.out_dir):
        os.mkdir(args.out_dir)

    evaluate(args)