```python
import argparse
from fate_arch.common import log
from fate_flow.entity.metric import Metric, MetricMeta
from fate_flow.components.component_base import ComponentBase
from fate_flow.components.hetero_nn.backend.keras.models import build_hetero_nn
from fate_flow.components.hetero_nn.backend.keras.layers import Dense
from fate_flow.components.hetero_nn.param import HeteroNNParam
from fate_flow.utils import data_io_utils
from fate_flow.utils.config_adapter import JobRuntimeConfigAdapter

class HeteroNNPipeline(ComponentBase):
    def __init__(self):
        super(HeteroNNPipeline, self).__init__()
        self.model = None

    def run(self, config_path):
        # Load configuration
        config = self.load_config(config_path)
        data_config = config["data"]
        model_config = config["model"]

        # Initialize data
        guest_data, guest_header = data_io_utils.load_data(data_config["guest"])
        host_data, host_header = data_io_utils.load_data(data_config["host"])

        # Data preprocessing
        guest_data, host_data = data_and_header_alignment(guest_data, guest_header, host_data, host_header)

        # Model configuration
        hetero_nn_param = HeteroNNParam(**model_config)
        self.model = build_hetero_nn(hetero_nn_param)

        # Model training
        self.model.fit(guest_data, host_data, epochs=hetero_nn_param.epochs, batch_size=hetero_nn_param.batch_size, callbacks=hetero_nn_param.callbacks)

        # Model evaluation
        eval_result = self.model.evaluate(guest_data, host_data)
        self.log_metric("evaluation", eval_result)

    def load_config(self, config_path):
        with open(config_path, 'r') as f:
            config = json.load(f)
        return config

    def log_metric(self, name, value):
        metric_meta = MetricMeta(name=name, metric_type="EVALUATION", extra_metas={})
        metric = Metric(name=name, metric_namespace="train", metric_data=[("value", value)])
        self.tracker.log_metric_data(metric_namespace=metric_meta.metric_namespace,
                                     metric_name=metric_meta.name,
                                     metrics=[metric])
        self.tracker.set_metric_meta(metric_namespace=metric_meta.metric_namespace,
                                     metric_name=metric_meta.name,
                                     metric_meta=metric_meta)

def main(config_path):
    pipeline = HeteroNNPipeline()
    pipeline.run(config_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config_path", type=str, help="Path to the configuration file")
    args = parser.parse_args()

    main(args.config_path)
```