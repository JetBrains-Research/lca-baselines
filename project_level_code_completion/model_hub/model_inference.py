import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
from tqdm.auto import tqdm

# from lca.code_generation.data_classes.datapoint_commit_dataset import DatapointCommitDataset
from model_hub.model_registry import MODEL_REGISTRY


def get_model(args):
    if args.model == "h3_pretrained_fl":
        raise NotImplementedError
    else:
        model_metainfo = MODEL_REGISTRY[args.model]
        model, device = model_metainfo.builder.build_model(model_metainfo.checkpoint, trust_remote_code=True)
    return model, device


def get_input_data(args):
    json_path = args.input_data_path
    with open(json_path, "r") as json_file:
        json_data = json.load(json_file)

    input_data = json_data.copy()  #[DatapointCommitDataset(**el) for el in json_data]

    return input_data


@torch.inference_mode()
def model_inference(
        model,
        device,
        input_data,
        seq_max_len,
        context_max,
        out_dir,
):
    ctxt_lens = list()
    repo_ids = list()
    crop_lens = list()
    input_lens = list()

    for num_dp, datapoint in enumerate(tqdm(input_data)):
        completion_len = len(datapoint['model_input']) - datapoint['context_len']  # initial len of `completion`
        if completion_len > seq_max_len / 4:
            last_idx = datapoint['context_len'] + seq_max_len // 4
            input_ids_cropped = datapoint['model_input'][:last_idx]
            completion_len = len(input_ids_cropped) - datapoint['context_len']
        else:
            input_ids_cropped = datapoint['model_input'].copy()
        input_ids_cropped = input_ids_cropped[-seq_max_len:]
        input_ids = torch.tensor(input_ids_cropped).unsqueeze(0).to(device)

        datapoint['context_len'] = max(len(input_ids_cropped) - completion_len, 0)
        context_len = datapoint['context_len']
        if context_len > context_max:
            thr = context_len - context_max
            input_ids = input_ids[..., thr:]
            context_len = context_max
        crop_len = len(datapoint['model_input']) - input_ids.size(-1)
        crop_lens.append(crop_len)
        input_lens.append(len(datapoint['model_input']))

        out = model(input_ids)

        logits = out.logits

        curr_dir = os.path.join(out_dir, f'repo_{datapoint["repo_id"]}_{num_dp}')
        # if not os.path.exists((curr_dir)):
        os.mkdir(curr_dir)

        # np.save(os.path.join(curr_dir, 'context_logits.npy'), logits[0].detach().cpu().numpy()[:context_len])
        np.save(os.path.join(curr_dir, 'completion_logits.npy'),
                logits[0].detach().cpu().to(torch.float32).numpy().astype(np.float16)[context_len:])
        np.save(os.path.join(curr_dir, 'context_tokens.npy'), input_ids[0].detach().cpu().numpy()[:context_len])
        np.save(os.path.join(curr_dir, 'completion_tokens.npy'), input_ids[0].detach().cpu().numpy()[context_len:])

        ctxt_lens.append(context_len)
        repo_ids.append(str(f"{datapoint['repo_id']}_{num_dp}"))

    with open(os.path.join(out_dir, 'context_lengths.json'), 'w') as json_file:
        json.dump(dict(zip(repo_ids, ctxt_lens)), json_file)
    with open(os.path.join(out_dir, 'input_lengths.json'), 'w') as json_file:
        json.dump(dict(zip(repo_ids, input_lens)), json_file)

    return {'lost_tokens_num': sum(crop_lens), 'lost_tokens_mean': sum(crop_lens)/len(crop_lens), 'lost_tokens_ratio': sum(crop_lens)/sum(input_lens)}


@torch.inference_mode()
def inference(args):
    if args.model == "h3_pretrained_fl":
        raise NotImplementedError

    else:
        model, device = get_model(args)
        input_data = get_input_data(args)

        lost_tokens = model_inference(model, device, input_data, seq_max_len=args.seq_max_len,
                                      context_max=args.context_max, out_dir=args.out_dir)

    return lost_tokens


if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        '--model', '-m',
        type=str,
        choices=list(MODEL_REGISTRY.keys()),
        help='Choose a model from the model hub'
    )
    argparser.add_argument(
        '--input_data_path', '-i',
        type=str,
        help='Path to json file with prepared data'
    )
    argparser.add_argument('--seq_max_len', '-s', type=int, default=4096, help='Maximal possible sequence length')
    argparser.add_argument('--context_max', '-c', type=int, default=2048, help='Maximal possible context length')
    argparser.add_argument('--out_dir', type=str, help='directory to save logits')

    args = argparser.parse_args()

    out_dir_path = Path(args.out_dir)
    out_dir_path.mkdir(parents=True, exist_ok=True)

    inference(args)

