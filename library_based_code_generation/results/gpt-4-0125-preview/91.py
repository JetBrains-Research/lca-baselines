```python
import matplotlib.pyplot as plt
import pandas as pd
from shiny import App, render, ui, reactive
import time

# Check if running in Pyodide environment
try:
    import pyodide
    IN_PYODIDE = True
except ImportError:
    IN_PYODide = False

# Fake psutil for Pyodide environment
class FakePsutil:
    @staticmethod
    def cpu_percent(interval=None, percpu=False):
        import random
        if percpu:
            return [random.randint(1, 100) for _ in range(4)]  # Assuming 4 cores
        else:
            return random.randint(1, 100)

# Use fake psutil in Pyodide, real psutil otherwise
if IN_PYODIDE:
    psutil = FakePsutil()
else:
    import psutil

def hide_ticks(ax, axis='both'):
    if axis in ('both', 'x'):
        ax.xaxis.set_major_locator(plt.NullLocator())
    if axis in ('both', 'y'):
        ax.yaxis.set_major_locator(plt.NullLocator())

app_ui = ui.page_fluid(
    ui.input_slider("num_samples", "Number of Samples per Graph", min=1, max=100, value=50),
    ui.input_slider("num_rows", "Number of Rows in Table", min=1, max=20, value=10),
    ui.input_select("colormap", "Select Colormap", {'viridis': 'viridis', 'plasma': 'plasma', 'inferno': 'inferno', 'magma': 'magma'}),
    ui.input_action_button("clear_history", "Clear History"),
    ui.input_action_button("freeze_output", "Freeze Output"),
    ui.output_plot("cpu_usage_plot"),
    ui.output_table("cpu_usage_table")
)

def server(input, output, session):
    cpu_history = reactive.Value(pd.DataFrame(columns=["Time", "CPU Usage"]))
    freeze_output = reactive.Value(False)

    @reactive.Effect
    @reactive.event(input.clear_history)
    def _():
        cpu_history.set(pd.DataFrame(columns=["Time", "CPU Usage"]))

    @reactive.Effect
    @reactive.event(input.freeze_output)
    def _():
        freeze_output.set(not freeze_output.get())

    @reactive.Effect
    @reactive.timer(interval=1, priority=reactive.TimerPriority.high)
    def _():
        if not freeze_output.get():
            current_time = time.strftime("%H:%M:%S")
            cpu_usage = psutil.cpu_percent(interval=1)
            new_data = pd.DataFrame([[current_time, cpu_usage]], columns=["Time", "CPU Usage"])
            history = cpu_history.get()
            cpu_history.set(history.append(new_data, ignore_index=True))

    @output
    @render.plot
    def cpu_usage_plot():
        df = cpu_history.get().tail(input.num_samples())
        fig, ax = plt.subplots()
        ax.plot(df["Time"], df["CPU Usage"], color='tab:blue')
        ax.set_xlabel('Time')
        ax.set_ylabel('CPU Usage (%)')
        ax.set_title('CPU Usage Over Time')
        plt.set_cmap(input.colormap())
        hide_ticks(ax, 'x')
        plt.xticks(rotation=45)
        plt.tight_layout()
        return fig

    @output
    @render.table
    def cpu_usage_table():
        df = cpu_history.get().tail(input.num_rows())
        return df

app = App(app_ui, server)

if __name__ == "__main__":
    app.run()
```