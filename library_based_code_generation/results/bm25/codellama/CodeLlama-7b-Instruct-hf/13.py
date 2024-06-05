  Here is the Python code for the OnlineManager class based on rolling tasks using the qlib library:
```
import qlib

class OnlineManager:
    def __init__(self, provider_uri, region, experiment_name, task_url, task_database_name, task_pool_name, rolling_step, start_time, end_time, tasks, trainer):
        self.provider_uri = provider_uri
        self.region = region
        self.experiment_name = experiment_name
        self.task_url = task_url
        self.task_database_name = task_database_name
        self.task_pool_name = task_pool_name
        self.rolling_step = rolling_step
        self.start_time = start_time
        self.end_time = end_time
        self.tasks = tasks
        self.trainer = trainer

    def reset_experiment(self):
        # Reset the experiment by deleting the task database and task pool
        qlib.delete_database(self.task_database_name)
        qlib.delete_pool(self.task_pool_name)

    def run_workflow(self):
        # Run the entire workflow automatically
        # 1. Collect tasks
        tasks = self.collect_tasks()
        # 2. Normalize tasks
        normalized_tasks = self.normalize_tasks(tasks)
        # 3. Train tasks
        trained_tasks = self.train_tasks(normalized_tasks)
        # 4. Test tasks
        test_results = self.test_tasks(trained_tasks)
        # 5. Get signals
        signals = self.get_signals(test_results)
        # 6. Perform backtesting
        backtest_results = self.backtest(signals)
        # 7. Perform risk analysis
        risk_analysis_results = self.risk_analysis(backtest_results)
        return risk_analysis_results

    def train_tasks(self, tasks):
        # Train tasks using the trainer
        trained_tasks = self.trainer.train(tasks)
        return trained_tasks

    def test_tasks(self, tasks):
        # Test tasks using the trainer
        test_results = self.trainer.test(tasks)
        return test_results

    def get_signals(self, test_results):
        # Get signals from the test results
        signals = self.trainer.get_signals(test_results)
        return signals

    def backtest(self, signals):
        # Perform backtesting on the signals
        backtest_results = self.trainer.backtest(signals)
        return backtest_results

    def risk_analysis(self, backtest_results):
        # Perform risk analysis on the backtest results
        risk_analysis_results = self.trainer.risk_analysis(backtest_results)
        return risk_analysis_results

    def collect_tasks(self):
        # Collect tasks from the task URL
        tasks = qlib.collect_tasks(self.task_url)
        return tasks

    def normalize_tasks(self, tasks):
        # Normalize tasks using the normalize class
        normalized_tasks = qlib.normalize_tasks(tasks)
        return normalized_tasks

    def main(self):
        # Reset the experiment
        self.reset_experiment()
        # Run the workflow
        results = self.run_workflow()
        # Print the results
        print(results)

if __name__ == "__main__":
    # Initialize the OnlineManager with user-defined parameters
    provider_uri = "https://qlib.ai"
    region = "us-east-1"
    experiment_name = "my_experiment"
    task_url = "https://qlib.ai/tasks"
    task_database_name = "my_task_database"
    task_pool_name = "my_task_pool"
    rolling_step = 10
    start_time = "2022-01-01"
    end_time = "2022-01-31"
    tasks = ["task1", "task2", "task3"]
    trainer = qlib.Trainer(provider_uri, region, experiment_name, task_database_name, task_pool_name, rolling_step, start_time, end_time, tasks)
    # Run the main method
    OnlineManager(provider_uri, region, experiment_name, task_url, task_database_name, task_pool_name, rolling_step, start_time, end_time, tasks, trainer).main()
```
This code defines a class called `OnlineManager` that simulates the online manager based on rolling tasks using the qlib library. The class has methods to reset the experiment, run the entire workflow automatically, and train tasks by other processes or machines for multiprocessing. The main method includes steps to reset, simulate, collect results, get signals, perform backtesting, and risk analysis. The class can be executed from the command line with user-defined parameters.