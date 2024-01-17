import argparse
import logging
import os
import torch

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="3"

from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from openai import OpenAI
from utils.files_utils import load_config
from utils.openai_generation import GPT_generation
from utils.context_utils import collect_good_context, trim_context
from tqdm.auto import tqdm

DEVICE = "cuda" if torch.cuda.is_available() else ""
def prepare_code_context(row, max_context_toks, tokenizer):
    context = collect_good_context(row)
    if max_context_toks is None:
        return context
    return trim_context(context, tokenizer, max_context_toks)

def generate_one(row, code_context, client, model_name, tokenizer, model):
    intent = row['instruction']
    intent_add = row['instruction_add']

    prompt = '[INST] I have code collected from one or more files joined into one string.'
    prompt += 'Using the code generate text to {intent}\n\n'
    prompt += f'My code: {code_context}'
    prompt += f'As answer return only documentation about {intent} [/INST]'

    tok_prompt = tokenizer(prompt, return_tensors='pt').to(DEVICE)
    with torch.no_grad():
        res = model.generate(**tok_prompt, max_length=2000)
    answer = tokenizer.batch_decode(res)[0]
    answer = answer.split('[/INST]')[1]
    return answer

def generate_all(config, client):

    # Extract parameters
    hf_api_key = config.get("hf_api_key")
    hf_tokenizer_checkpoint = config.get("hf_tokenizer_checkpoint", "bigcode/starcoder")
    model_name = config.get("model_name", "gpt-3.5-turbo-16k")
    max_context_toks = config.get("max_context_toks", None)
    save_dir = config.get("save_dir", "./predictions")
    if not os.path.exists(f"{save_dir}"):
        os.makedirs(f"{save_dir}")

    logging.info("Downloading dataset")
    # Preparing dataset
    dataset = load_dataset("JetBrains-Research/lca-module-to-text",
                           token=hf_api_key)['train']
    logging.info("Downloading tokenizer and model")
    tokenizer = AutoTokenizer.from_pretrained(hf_tokenizer_checkpoint, token=hf_api_key)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        token=hf_api_key,
        load_in_8bit=True,
        torch_dtype=torch.float16,
        device_map="auto"
    ).eval()

    logging.info("Start generation process")
    # Generation
    for row_idx, row in tqdm(enumerate(dataset), total=len(dataset), position=0, leave=True, desc="Generation"):
        try:
            code_context = prepare_code_context(row, max_context_toks, tokenizer)
            generate_res = generate_one(row, code_context, client, model_name, tokenizer, model)

            # Saving prediction
            with open(f"{save_dir}/{row_idx}.txt", 'w') as f:
                f.write(generate_res)
        except Exception as e:
            logging.error(str(e))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Script with YAML config and command line arguments.")
    # Argument for YAML config file path
    parser.add_argument('--config', type=str, default="config.yaml", help='Path to the YAML configuration file')
    args = parser.parse_args()

    config = load_config(args.config)
    logs_dir = config.get("logs_dir", "./logs")
    if not os.path.exists(f"{logs_dir}"):
        os.makedirs(f"{logs_dir}")
    openai_api_key = config.get("openai_api_key")
    model_name = config.get("model_name", "gpt-3.5-turbo-16k")

    mn = model_name.split('/')[-1]
    logging.basicConfig(
        filename=f'{logs_dir}/chatgpt_gen_{mn}.log',
        encoding='utf-8',
        level=logging.DEBUG
    )

    logging.info("Creating OpenAI client")
    client = OpenAI(api_key=openai_api_key)
    logging.info("Done")

    logging.info("Call generate all function")
    generate_all(config, client)
    logging.info("Work finished")
