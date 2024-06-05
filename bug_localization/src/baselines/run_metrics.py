import ast
import os

import hydra
import pandas as pd
from dotenv import load_dotenv

from src.baselines.configs.baseline_configs import BaselineConfig
from src.baselines.data_sources.base_data_source import BaseDataSource


@hydra.main(version_base="1.1", config_path="../../configs/baselines")
def main(cfg: BaselineConfig) -> None:
    data_src: BaseDataSource = hydra.utils.instantiate(cfg.data_src)
    results_path = os.path.join(cfg.output_path, cfg.name)
    os.makedirs(results_path, exist_ok=True)
    results_csv_path = os.path.join(results_path, "results.csv")
    df = pd.read_csv(results_csv_path)

    index = 0
    for dp, repo_content in data_src:
        expected_files = ast.literal_eval(df.iloc[index]['expected_files'])
        changed_files = ast.literal_eval(dp['changed_files'])
        print('Predicted:', expected_files)
        print('Changed:', changed_files)
        print('Common:', set(expected_files).intersection(changed_files))
        index += 1


if __name__ == '__main__':
    load_dotenv()
    main()
