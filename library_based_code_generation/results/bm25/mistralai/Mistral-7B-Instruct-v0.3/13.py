 Here is a basic structure for the class using the qlib library. Please note that this is a simplified version and might need adjustments based on your specific requirements.

```python
import sys
import time
import qlib
import qlib.qlib as qt
import qlib.data.builder as qdb
import qlib.ts as qts
import qlib.baselines as qbl
import qlib.backtest as qbt
import qlib.risk_analysis as qra
import qlib.utils as qtu
import concurrent.futures

class OnlineManager:
    def __init__(self, provider_uri, region, experiment_name, task_url, task_db_name, task_pool_name, rolling_step, start_time, end_time, tasks, trainer):
        self.provider_uri = provider_uri
        self.region = region
        self.experiment_name = experiment_name
        self.task_url = task_url
        self.task_db_name = task_db_name
        self.task_pool_name = task_pool_name
        self.rolling_step = rolling_step
        self.start_time = start_time
        self.end_time = end_time
        self.tasks = tasks
        self.trainer = trainer
        self.collector_class_name = "Collector"
        self.normalize_class_name = "Normalize"

    def reset(self):
        qt.reset_parameters()

    def run(self):
        qt.setUpClass()
        self.reset()
        self.simulate()
        qt.tearDownClass()

    def simulate(self):
        collector = qt.class_casting(self.collector_class_name)
        normalize = qt.class_casting(self.normalize_class_name)

        with concurrent.futures.ProcessPoolExecutor() as executor:
            tasks = [(executor.submit, (self._train_rolling_tasks, collector, normalize, task, self.trainer)) for task in self.tasks]
            for _ in executor.map(_and__, tasks):
                pass

    def _train_rolling_tasks(self, collector, normalize, task, trainer):
        # Implement task training logic here
        pass

    def collect_results(self):
        # Implement result collection logic here
        pass

    def get_signals(self):
        # Implement signal extraction logic here
        pass

    def perform_backtesting(self):
        # Implement backtesting logic here
        pass

    def risk_analysis(self):
        # Implement risk analysis logic here
        pass

    def main(self):
        self.run()
        self.collect_results()
        self.get_signals()
        self.perform_backtesting()
        self.risk_analysis()

if __name__ == "__main__":
    manager = OnlineManager(
        provider_uri="your_provider_uri",
        region="your_region",
        experiment_name="your_experiment_name",
        task_url="your_task_url",
        task_db_name="your_task_db_name",
        task_pool_name="your_task_pool_name",
        rolling_step=1,
        start_time=time.time(),
        end_time=time.time() + 86400,  # 1 day
        tasks=[],
        trainer=None,
    )
    manager.main()
```

This code creates a class named `OnlineManager` that initializes with the specified parameters. The class has methods for resetting the experiment, running the workflow, simulating tasks, collecting results, getting signals, performing backtesting, and risk analysis. The main method is executed when the script is run from the command line.

Please replace the placeholders with your actual provider URI, region, experiment name, task URL, task database name, task pool name, and other parameters. Also, you should implement the missing methods according to your specific requirements.

To run the script from the command line, save it as `online_manager.py` and execute it with the following command:

```
python online_manager.py
```