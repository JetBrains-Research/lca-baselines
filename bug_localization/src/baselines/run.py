import json
import os
import time
from pathlib import Path

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

from src import PROJECT_DIR
from src.baselines.backbone.base_backbone import BaseBackbone
from src.baselines.data_sources.base_data_source import BaseDataSource


@hydra.main(version_base="1.1", config_path=os.path.join(PROJECT_DIR / "configs"), config_name="run.yaml")
def main(cfg: DictConfig) -> None:
    OmegaConf.set_struct(cfg, False)
    os.environ['HYDRA_FULL_ERROR'] = '1'

    backbone: BaseBackbone = hydra.utils.instantiate(cfg.backbone)
    data_src: BaseDataSource = hydra.utils.instantiate(cfg.data_source)

    results_dir_path = Path(HydraConfig.get().run.dir)
    os.makedirs(results_dir_path, exist_ok=True)
    results_path = results_dir_path / 'results.jsonl'

    for dp in data_src:
        start_time = time.time()
        results_dict = backbone.localize_bugs(dp)
        end_time = time.time()
        results_dict['time_s'] = (end_time - start_time) * 1000000

        with open(results_path, 'a', newline='') as f:
            f.write(json.dumps(results_dict) + "\n")


if __name__ == '__main__':
    main()
