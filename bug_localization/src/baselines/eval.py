import json
import os
from pathlib import Path

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

from src import PROJECT_DIR, logger
from src.baselines.data_sources.base_data_source import BaseDataSource
from src.baselines.metrics.context_metrics import get_context_metrics
from src.baselines.metrics.quality_metrics import get_quality_metrics


@hydra.main(version_base="1.1", config_path=os.path.join(PROJECT_DIR / "configs"), config_name="eval.yaml")
def main(cfg: DictConfig) -> None:
    OmegaConf.set_struct(cfg, False)
    os.environ['HYDRA_FULL_ERROR'] = '1'

    data_src: BaseDataSource = hydra.utils.instantiate(cfg.data_source)

    eval_dir_path = Path(HydraConfig.get().run.dir)
    os.makedirs(eval_dir_path, exist_ok=True)
    eval_results_path = eval_dir_path / 'results.jsonl'

    run_results_path = os.path.join(cfg.data_path, 'run', cfg.run_id, 'results.jsonl')
    run_by_text_id = {}
    with open(run_results_path, 'r') as f:
        for line in f.readlines():
            data = json.loads(line.strip())
            text_id = data.get('text_id')
            run_by_text_id[text_id] = data

    for dp in data_src:
        run_result = run_by_text_id.get(dp['text_id'])
        if not run_result:
            continue
        if run_result['json_completion']:
            try:
                files = json.loads(run_result['json_completion'])
            except Exception as e:
                files = {'files': []}

            logger.info(files)
            logger.info(len(files['files']))

            all_files = [path for path, _ in dp['repo_content'].items()]
            expected_files = dp['changed_files']
            actual_files = files['files'] if files else []

            eval_result = {'text_id': dp['text_id']}

            quality_metrics = get_quality_metrics(all_files, expected_files, actual_files)
            logger.info(quality_metrics)
            eval_result.update(quality_metrics)

            messages = json.loads(run_result['messages'])
            context_metrics = get_context_metrics(messages)
            eval_result.update(context_metrics)

            with open(eval_results_path, 'a', newline='') as f:
                f.write(json.dumps(eval_result) + "\n")


if __name__ == '__main__':
    main()
