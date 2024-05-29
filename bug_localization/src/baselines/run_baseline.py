import csv
import os

import hydra
from dotenv import load_dotenv
from hydra.core.hydra_config import HydraConfig

from src.baselines.backbones.base_backbone import BaseBackbone
from src.baselines.configs.baseline_configs import BaselineConfig
from src.baselines.data_sources.base_data_source import BaseDataSource


@hydra.main(version_base="1.1", config_path="../../configs/baselines")
def main(cfg: BaselineConfig) -> None:
    os.environ['HYDRA_FULL_ERROR'] = '1'
    backbone: BaseBackbone = hydra.utils.instantiate(cfg.backbone)
    data_src: BaseDataSource = hydra.utils.instantiate(cfg.data_source)

    output_path = HydraConfig.get().run.dir
    os.makedirs(output_path, exist_ok=True)
    results_csv_path = os.path.join(output_path, "results.csv")

    count = 3
    for dp, repo_content, changed_files in data_src:
        if count == 0:
            break
        count -= 1
        issue_description = dp['issue_title'] + '\n' + dp['issue_body']
        results_dict = backbone.localize_bugs(issue_description, repo_content)
        results_dict['text_id'] = dp['text_id']

        with open(results_csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            if f.tell() == 0:
                writer.writerow(results_dict.keys())
            writer.writerow(results_dict.values())


if __name__ == '__main__':
    load_dotenv()
    main()
