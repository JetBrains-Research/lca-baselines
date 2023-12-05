import os
from argparse import ArgumentParser

from omegaconf import DictConfig, OmegaConf

from baselines.baseline_models import ScoreBaseline, EmbedBaseline
from baselines.baseline_tokenizers import BaseTokenizer
from baselines.models.codet5_baseline import CodeT5Baseline
from baselines.models.openai_baseline import OpenAIBaseline
from baselines.models.tf_idf_baseline import TfIdfBaseline
from baselines.tokenizers.bpe_tokenizer import BPETokenizer
from baselines.tokenizers.codet5_tokenizer import CodeT5Tokenizer
from baselines.tokenizers.nltk_tokenizer import NltkTokenizer
from run_embed_baseline import run_embed_baseline
from run_score_baseline import run_score_baseline


def init_score_baseline(config: DictConfig) -> ScoreBaseline:
    if config.model.baseline_name == OpenAIBaseline.name():
        return OpenAIBaseline(
            api_key=os.environ['OPENAI_API_KEY'],
            model=config.model.model
        )
    else:
        # Add your scoring baseline initialization here
        raise Exception(f"Baseline {config.baseline_name} is not supported")


def init_embed_tokenizer(config: DictConfig) -> BaseTokenizer:
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


def init_embed_baseline(config: DictConfig, pretrained_path: str) -> EmbedBaseline:
    if config.model.name == TfIdfBaseline.name():
        return TfIdfBaseline(
            pretrained_path=pretrained_path,
            tokenizer=init_embed_tokenizer(config),
        )
    if config.model.name == CodeT5Baseline.name():
        return CodeT5Baseline(
            pretrained_path=pretrained_path,
            device=config.model.device,
            checkpoint=config.model.checkpoint,
        )
    else:
        # Add your embed baseline initialization here
        raise Exception(f"Baseline {config.baseline_name} is not supported")


def get_run_directory(baseline_results_path: str) -> str:
    run_index = 0
    while os.path.exists(os.path.join(baseline_results_path, f'run_{run_index}')):
        run_index += 1

    run_path = os.path.join(baseline_results_path, f'run_{run_index}')
    os.makedirs(run_path, exist_ok=True)

    return run_path


def run_baseline(baseline_config_path: str, bug_localization_data_path: str):
    baseline_config = OmegaConf.load(baseline_config_path)
    results_path = os.path.join(bug_localization_data_path, 'results')
    if not os.path.exists(results_path):
        os.makedirs(results_path, exist_ok=True)

    run_path = get_run_directory(results_path)

    pretrained_path = os.path.join(bug_localization_data_path, 'pretrained')
    if not os.path.exists(pretrained_path):
        os.makedirs(pretrained_path, exist_ok=True)

    if baseline_config.baseline_type == 'embed':
        baseline = init_embed_baseline(baseline_config, pretrained_path)
        run_embed_baseline(baseline, run_path)
    else:
        baseline = init_score_baseline(baseline_config)
        run_score_baseline(baseline, run_path)


if __name__ == '__main__':
    argparser = ArgumentParser()

    argparser.add_argument(
        "--baseline-config-path",
        type=str,
        help="Path to yaml file with baseline model configuration.",
        default="./baselines/configs/tfidf_config.yaml"
    )

    argparser.add_argument(
        "--bug-localization-data-path",
        type=str,
        help="Path to directory where repos are stored.",
        default="./../data/lca-bug-localization"
    )

    argparser.add_argument(
        "--bug-localization-data-path",
        type=str,
        help="Path to directory where repos are stored.",
        default="./../data/lca-bug-localization"
    )

    args = argparser.parse_args()
