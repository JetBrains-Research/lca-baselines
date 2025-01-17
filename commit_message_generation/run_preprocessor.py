import hydra
import jsonlines
from dotenv import load_dotenv
from omegaconf import OmegaConf
from tqdm import tqdm  # type: ignore[import-untyped]

from configs import BaselineConfig
from configs.preprocessor_configs import BasePreprocessorConfig
from src import CMGPreprocessor

load_dotenv()


def init_preprocessor(cfg: BasePreprocessorConfig) -> CMGPreprocessor:
    if "local_data_dir" in cfg:  # type: ignore[operator]
        cfg.local_data_dir = hydra.utils.to_absolute_path(cfg.local_data_dir)  # type: ignore[attr-defined]
    preprocessor = hydra.utils.instantiate(cfg)
    return preprocessor


def get_predictions(
    preprocessor: CMGPreprocessor, cfg: BaselineConfig, predictions_path: str = "predictions.jsonl"
) -> str:
    # init iterator (either over local file or over HuggingFace dataset)
    if hasattr(cfg.data_src, "path"):
        cfg.data_src.path = hydra.utils.to_absolute_path(cfg.data_src.path)  # type: ignore[attr-defined]
    reader = hydra.utils.instantiate(cfg.data_src)

    # get predictions for all input examples
    open(predictions_path, "w").close()
    for line in tqdm(reader, "Building inputs"):
        baseline_output = preprocessor(line)
        cur_example = {"hash": line["hash"], "repo": line["repo"]}
        cur_example.update(baseline_output)

        with jsonlines.open(predictions_path, "a") as writer:
            writer.write(cur_example)
    return predictions_path


@hydra.main(version_base="1.1", config_path="configs", config_name="baseline_config")
def main(cfg: BaselineConfig) -> None:
    with open("config.yaml", "w") as f:
        OmegaConf.save(cfg, f)

    # init baseline
    preprocessor = init_preprocessor(cfg.preprocessor)

    # obtain predictions
    get_predictions(cfg=cfg, preprocessor=preprocessor)


if __name__ == "__main__":
    main()
