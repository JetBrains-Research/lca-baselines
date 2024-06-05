```python
import matplotlib.pyplot as plt
import pandas as pd
from shiny import App, ui, reactive, render, req
import sys

# Check if running in Pyodide environment
is_pyodide = sys.platform == "emscripten"

if is_pyodide:
    class FakePsutil:
        def cpu_percent(interval=None, percpu=False):
            # Generate fake CPU usage data
            import random
            if percpu:
                return [random.randint(1, 100) for _ in range(4)]  # Assuming 4 cores
            else:
                return random.randint(1, 100)
    psutil = FakePsutil
else:
    import psutil

def hide_ticks(ax, axis='x'):
    if axis == 'x':
        ax.xaxis.set_ticks_position('none') 
        ax.xaxis.set_ticklabels([])
    elif axis == 'y':
        ax.yaxis.set_ticks_position('none') 
        ax.yaxis.set_ticklabels([])

app_ui = ui.page_fluid(
    ui.input_action_button("refresh", "Refresh Data"),
    ui.input_action_button("clear", "Clear History"),
    ui.input_action_button("freeze", "Freeze/Unfreeze Output"),
    ui.input_slider("samples", "Number of Samples", min=1, max=100, value=10),
    ui.input_slider("rows", "Number of Rows in Table", min=1, max=20, value=5),
    ui.input_select("colormap", "Select Colormap", {"viridis": "viridis", "plasma": "plasma", "inferno": "inferno", "magma": "magma"}),
    ui.output_plot("cpu_plot"),
    ui.output_table("cpu_table")
)

def server(input, output, session):
    cpu_data = reactive.Value(pd.DataFrame(columns=["Time", "CPU Usage"]))
    frozen = reactive.Value(False)

    @reactive.Effect
    @reactive.event(input.refresh)
    def _():
        if frozen():
            return
        current_time = pd.Timestamp.now()
        cpu_usage = psutil.cpu_percent(interval=1)
        new_data = pd.DataFrame([[current_time, cpu_usage]], columns=["Time", "CPU Usage"])
        cpu_data(cpu_data().append(new_data, ignore_index=True))

    @reactive.Effect
    @reactive.event(input.clear)
    def _():
        cpu_data(pd.DataFrame(columns=["Time", "CPU Usage"]))

    @reactive.Effect
    @reactive.event(input.freeze)
    def _():
        frozen(not frozen())

    @output
    @render.plot
    def cpu_plot():
        req(cpu_data().shape[0] > 0)
        df = cpu_data().tail(input.samples())
        fig, ax = plt.subplots()
        ax.plot(df["Time"], df["CPU Usage"], color="tab:blue")
        ax.set_xlabel("Time")
        ax.set_ylabel("CPU Usage (%)")
        plt.set_cmap(input.colormap())
        hide_ticks(ax, 'x')
        return fig

    @output
    @render.table
    def cpu_table():
        df = cpu_data().tail(input.rows())
        return df

app = App(app_ui, server)

if __name__ == "__main__":
    app.run()
```