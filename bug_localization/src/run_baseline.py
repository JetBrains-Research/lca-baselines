import os

import pandas as pd
from omegaconf import DictConfig, OmegaConf

from src.baselines.model.baseline_models import Baseline
from src.baselines.model.baseline_tokenizers import BaseTokenizer
from src.baselines.backbones.codet5_embed_backbone import CodeT5Baseline
from src.baselines.backbones.openai_list_files_backbone import OpenAIBaseline
from src.baselines.backbones.tf_idf_backbone import TfIdfBaseline
from src.baselines.tokenizers.bpe_tokenizer import BPETokenizer
from src.baselines.tokenizers.codet5_tokenizer import CodeT5Tokenizer
from src.baselines.tokenizers.nltk_tokenizer import NltkTokenizer
from src.utils.file_utils import create_dir, create_run_directory, save_config
from src.utils.hf_utils import load_data


def init_tokenizer(config: DictConfig) -> BaseTokenizer:
    if config.model.tokenizer.name == NltkTokenizer.name():
        return NltkTokenizer()
    if config.model.tokenizer.name == CodeT5Tokenizer.name():
        return CodeT5Tokenizer(
            checkpoint=config.model.tokenizer.checkpoint,
        )
    if config.model.tokenizer.name == BPETokenizer.name():
        return BPETokenizer(
            pretrained_path=config.model.tokenizer.pretrained_path,
            vocab_size=config.model.tokenizer.vocab_size,
            min_frequency=config.model.tokenizer.min_frequency,
        )
    else:
        # Add your tokenizer initialization here
        raise Exception(f"Tokenizer {config.model.tokenizer.name} is not supported")


def init_model(config: DictConfig) -> Baseline:
    if config.model.name == OpenAIBaseline.name():
        return OpenAIBaseline(
            api_key=os.environ['OPENAI_API_KEY'],
            model=config.model.model
        )
    if config.model.name == TfIdfBaseline.name():
        return TfIdfBaseline(
            repos_path=config.repos_path,
            pretrained_path=config.pretrained_path,
            tokenizer=init_tokenizer(config),
        )
    if config.model.name == CodeT5Baseline.name():
        return CodeT5Baseline(
            pretrained_path=config.pretrained_path,
            device=config.model.device,
            checkpoint=config.model.checkpoint,
        )
    else:
        # Add your embed baseline initialization here
        raise Exception(f"Baseline {config.baseline_name} is not supported")


def run_baseline() -> None:
    run_config = OmegaConf.load("../configs/run.yaml")
    local_config = OmegaConf.load(f"../configs/data/{run_config.data}.yaml")
    baseline_config = OmegaConf.load(f"../configs/baselines/{run_config.baseline}.yaml")
    config = OmegaConf.merge(run_config, local_config, baseline_config)

    run_path, run_index = create_run_directory(os.path.join(config.data_path, 'runs'))
    save_config(config, run_path)

    for category in config.categories:
        for split in config.splits:
            df = load_data(category, split)
            config['results_path'] = create_dir(os.path.join(run_path, category, split))
            config['pretrained_path'] = create_dir(os.path.join(config.data_path, 'pretrained', category, split))
            model = init_model(config)

            metrics_list = model.run(df, category, split)
            pd.DataFrame([metrics.to_dict() for metrics in metrics_list]).to_csv(
                os.path.join(config['results_path'], 'metrics.csv'), index=False)


if __name__ == '__main__':
    run_baseline()
