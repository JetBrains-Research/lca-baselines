```python
import argparse
import qlib
from qlib.workflow import R
from qlib.workflow.online.update import run_online_task
from qlib.workflow.task.manage import TaskManager
from qlib.workflow.task.gen import RollingGen
from qlib.workflow.task.utils import TimeAdjuster
from qlib.workflow import OnlineManager
from qlib.config import REG_CN
from qlib.utils import init_instance_by_config

class OnlineManagerSimulator:
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
        self.online_manager = None
        self._init_qlib()

    def _init_qlib(self):
        qlib.init(provider_uri=self.provider_uri, region=self.region)

    def reset_experiment(self):
        TaskManager(task_url=self.task_url).remove_task_pool(self.task_pool_name)
        self.online_manager = OnlineManager(experiment_name=self.experiment_name)

    def run_workflow(self):
        rolling_gen = RollingGen(
            step=self.rolling_step,
            start_time=self.start_time,
            end_time=self.end_time,
            tasks=self.tasks,
            experiment_name=self.experiment_name,
            task_pool_name=self.task_pool_name,
            trainer=self.trainer
        )
        rolling_gen.gen()

    def train_tasks(self):
        task_manager = TaskManager(task_url=self.task_url, task_db_name=self.task_db_name)
        for task in task_manager.query_task({'task_pool': self.task_pool_name}):
            run_online_task(task, experiment_name=self.experiment_name)

    @staticmethod
    def main():
        parser = argparse.ArgumentParser()
        parser.add_argument("--provider_uri", type=str, required=True)
        parser.add_argument("--region", type=str, default=REG_CN)
        parser.add_argument("--experiment_name", type=str, required=True)
        parser.add_argument("--task_url", type=str, required=True)
        parser.add_argument("--task_db_name", type=str, required=True)
        parser.add_argument("--task_pool_name", type=str, required=True)
        parser.add_argument("--rolling_step", type=int, required=True)
        parser.add_argument("--start_time", type=str, required=True)
        parser.add_argument("--end_time", type=str, required=True)
        parser.add_argument("--tasks", type=str, required=True)  # Assuming JSON string for simplicity
        parser.add_argument("--trainer", type=str, required=True)  # Assuming JSON string for simplicity

        args = parser.parse_args()

        tasks = eval(args.tasks)
        trainer = eval(args.trainer)

        simulator = OnlineManagerSimulator(
            provider_uri=args.provider_uri,
            region=args.region,
            experiment_name=args.experiment_name,
            task_url=args.task_url,
            task_db_name=args.task_db_name,
            task_pool_name=args.task_pool_name,
            rolling_step=args.rolling_step,
            start_time=args.start_time,
            end_time=args.end_time,
            tasks=tasks,
            trainer=trainer
        )

        simulator.reset_experiment()
        simulator.run_workflow()
        simulator.train_tasks()
        # Steps for simulate, collect results, get signals, perform backtesting, and risk analysis would go here
        # This is a simplified example and does not include the implementation of these steps

if __name__ == "__main__":
    OnlineManagerSimulator.main()
```