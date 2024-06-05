  import psutil
import py_shiny
import matplotlib.pyplot as plt
import pandas as pd

class CPUUsageMonitor:
    def __init__(self):
        self.cpu_usage_data = []
        self.cpu_usage_history = []
        self.num_samples = 10
        self.num_rows = 10
        self.colormap = 'viridis'
        self.frozen = False

    def get_cpu_usage(self):
        if py_shiny.is_pyodide():
            return psutil.cpu_percent(interval=1, percpu=False)
        else:
            return psutil.cpu_percent(interval=1, percpu=True)

    def update_cpu_usage_data(self):
        self.cpu_usage_data.append(self.get_cpu_usage())
        if len(self.cpu_usage_data) > self.num_samples:
            self.cpu_usage_data = self.cpu_usage_data[-self.num_samples:]

    def update_cpu_usage_history(self):
        self.cpu_usage_history.append(self.cpu_usage_data)
        if len(self.cpu_usage_history) > self.num_rows:
            self.cpu_usage_history = self.cpu_usage_history[-self.num_rows:]

    def display_cpu_usage_graph(self):
        plt.plot(self.cpu_usage_data)
        plt.xlabel('Time')
        plt.ylabel('CPU Usage (%)')
        plt.title('CPU Usage Over Time')
        plt.colormap(self.colormap)
        plt.show()

    def display_cpu_usage_table(self):
        df = pd.DataFrame(self.cpu_usage_history)
        df.columns = ['CPU Usage (%)']
        df.index = pd.Index(range(len(self.cpu_usage_history)), name='Time')
        print(df)

    def clear_cpu_usage_history(self):
        self.cpu_usage_history = []

    def freeze_output(self):
        self.frozen = True

    def hide_ticks(self, axis):
        plt.tick_params(axis=axis, which='both', bottom=False, top=False, labelbottom=False)

if __name__ == '__main__':
    monitor = CPUUsageMonitor()
    while True:
        monitor.update_cpu_usage_data()
        monitor.update_cpu_usage_history()
        monitor.display_cpu_usage_graph()
        monitor.display_cpu_usage_table()
        if monitor.frozen:
            break
