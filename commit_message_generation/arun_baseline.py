import asyncio
import json
import logging
import os
import random
import sys
from argparse import ArgumentParser
from typing import Any, Dict

import hydra
import jsonlines
import pandas as pd  # type: ignore[import-untyped]
import wandb
from dotenv import load_dotenv
from hydra import compose, initialize
from omegaconf import OmegaConf
from tqdm import tqdm  # type: ignore[import-untyped]

from configs import BaselineConfig
from src import CMGBackbone, CMGBaseline, CMGMetrics

load_dotenv()

root = logging.getLogger()
root.setLevel(logging.INFO)
handler = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter("[%(asctime)s][%(name)s][%(levelname)s] - %(message)s")
handler.setFormatter(formatter)
root.addHandler(handler)


def init_baseline(cfg: BaselineConfig) -> CMGBaseline:
    # init backbone
    backbone: CMGBackbone = hydra.utils.instantiate(cfg.backbone)

    # init preprocessor
    preprocessor = hydra.utils.instantiate(
        cfg.preprocessor, model_name=cfg.backbone.model_name, model_provider=backbone.name
    )

    return CMGBaseline(backbone=backbone, preprocessor=preprocessor)


async def get_predictions(
    baseline: CMGBaseline, cfg: BaselineConfig, predictions_path: str = "predictions.jsonl"
) -> str:
    # init iterator (either over local file or over HuggingFace dataset)
    if hasattr(cfg.data_src, "path"):
        cfg.data_src.path = hydra.utils.to_absolute_path(cfg.data_src.path)  # type: ignore[attr-defined]
    reader = hydra.utils.instantiate(cfg.data_src)

    async def _get_prediction(line: Dict[str, Any]) -> None:
        baseline_output = await baseline.agenerate_msg(commit=line)  # type: ignore[arg-type]
        cur_example = {"reference": line["message"], "hash": line["hash"], "repo": line["repo"]}
        cur_example.update(baseline_output)

        with jsonlines.open(predictions_path, "a") as writer:
            writer.write(cur_example)

        return None

    # get predictions for all input examples
    open(predictions_path, "w").close()
    tasks = [_get_prediction(line) for line in reader]
    await asyncio.gather(*tasks)
    return predictions_path


def compute_metrics(predictions_path: str) -> Dict[str, float]:
    metrics = CMGMetrics()
    with jsonlines.open(predictions_path, "r") as reader:
        for example in tqdm(reader, desc="Computing metrics"):
            metrics.update(predictions=[example["prediction"]], references=[example["reference"]])
    computed_metrics = metrics.compute()
    print("=== METRICS ===")
    print(computed_metrics)
    return computed_metrics


async def main(config_name: str) -> None:
    initialize(version_base="1.1", config_path="configs/async")
    cfg_dict = compose(config_name=config_name)
    cfg = BaselineConfig(**cfg_dict)  # type: ignore

    os.makedirs(f"results/{config_name[: -len('.yaml')]}", exist_ok=True)

    if hasattr(cfg.backbone, "seed") and cfg.backbone.seed is None:
        cfg.backbone.seed = random.randint(1, 2**32)
        logging.warning(f"Using random seed {cfg.backbone.seed}.")

    # init W&B (optional)
    if cfg.logger.use_wandb:
        wandb.init(
            project=cfg.logger.project,
            name=cfg.logger.name,
            config=OmegaConf.to_container(cfg, resolve=True),  # type: ignore[arg-type]
            job_type="eval",
        )

    # init baseline
    baseline = init_baseline(cfg)

    # obtain predictions
    predictions_path = await get_predictions(
        cfg=cfg, baseline=baseline, predictions_path=f"results/{config_name[: -len('.yaml')]}/predictions.jsonl"
    )

    # log predictions to W&B (optional)
    if cfg.logger.use_wandb:
        artifact = wandb.Artifact(
            f"{cfg.backbone.model_name.replace('/', '__')}_{cfg.preprocessor._target_.split('.')[-1]}_{cfg.logger.name + '_' if cfg.logger.name else ''}predictions",
            type="dataset",
        )
        if cfg.logger.local_artifact:
            artifact.add_reference(f"file:///{os.path.abspath(predictions_path)}")
        else:
            test_table = wandb.Table(dataframe=pd.read_json(predictions_path, orient="records", lines=True))
            artifact.add(test_table, "predictions")
        wandb.log_artifact(artifact)

    # compute metrics
    computed_metrics = compute_metrics(predictions_path)
    with open(f"results/{config_name[: -len('.yaml')]}/metrics.json", "w") as f:
        json.dump(computed_metrics, f)

    # log metrics to W&B (optional)
    if cfg.logger.use_wandb:
        wandb.log(computed_metrics)


if __name__ == "__main__":
    parser = ArgumentParser(
        description="Launch a commit message generation model for BenchName dataset asynchronously."
    )
    parser.add_argument(
        "--config-name", type=str, help="Which config under `configs/async` directory to use.", required=True
    )
    args = parser.parse_args()

    asyncio.run(main(args.config_name))
