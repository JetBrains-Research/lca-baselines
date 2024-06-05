import datasets
import hydra
from dotenv import load_dotenv
from omegaconf import DictConfig
import shutil
from datasets import config
from src.utils.hf_utils import update_hf_data_splits


def split_data(df: datasets.Dataset, split: str, test_data_ids: list[str]):
    test_data_ids = [i.lower() for i in test_data_ids]
    if split == 'dev':
        return df

    if split == 'test':
        return df.filter(lambda dp: dp['text_id'].lower() in test_data_ids)

    if split == 'train':
        return df.filter(lambda dp: dp['text_id'].lower() not in test_data_ids)


@hydra.main(config_path="../../../configs/data", config_name="server", version_base=None)
def run_split_data(config: DictConfig):
    update_hf_data_splits(
        lambda df, category, split: split_data(df, split, config.test_data_ids),
    )


if __name__ == '__main__':
    cache_dir = config.HF_DATASETS_CACHE
    shutil.rmtree(cache_dir, ignore_errors=True)
    load_dotenv()
    run_split_data()
