 ```python
import asyncio
import psutil
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pyodide.http import pyodide_fetch
from bokeh.models import ColumnDataSource, Slider, Select
from bokeh.plotting import figure, show, output_notebook
from shiny import App, Inputs, Outputs, reactive, render, ui

# Fake psutil for Pyodide
if "pyodide" in __name__:
    class FakePsutil:
        def cpu_percent(self, interval=1):
            return 0

    psutil = FakePsutil()

async def get_cpu_data():
    data = {"time": [], "cpu_percent": []}
    while True:
        data["time"].append(time.time())
        data["cpu_percent"].append(psutil.cpu_percent())
        await asyncio.sleep(0.1)
        return data

def plot_cpu_usage(data):
    df = pd.DataFrame(data)
    source = ColumnDataSource(df)
    p = figure(x_axis_type="datetime", height=300, width=800, sizing_mode="stretch_both")
    p.line(x="time", y="cpu_percent", source=source, line_width=2, alpha=0.8, legend_label="CPU Usage")
    return p

def update_plot(plot, data):
    df = pd.DataFrame(data)
    source = ColumnDataSource(df)
    plot.title.text = f"CPU Usage (last {len(df)} samples)"
    plot.yaxis.axis_label = "Percentage"
    plot.xaxis.axis_label = "Time"
    plot.xaxis.major_label_overrides = {
        dt: f"{dt.strftime('%H:%M:%S')}" for dt in df["time"][::10]
    }
    plot.y_range.start = 0
    plot.y_range.end = 100
    plot.x_range.end = df["time"][-1]
    plot.x_range.start = df["time"][0]
    plot.patch.data_source = source

def hide_ticks(plot, axis="both"):
    if axis == "both":
        plot.xaxis.visible = False
        plot.yaxis.visible = False
    elif axis == "x":
        plot.xaxis.visible = False
    elif axis == "y":
        plot.yaxis.visible = False

app_ui = ui.page_fluid(
    ui.row(
        ui.column(
            6,
            ui.h3("CPU Usage"),
            ui.plot_output("plot"),
            ui.slider("samples", "Samples per graph", 10, 100, value=50, step=10, input_type="number"),
            ui.slider(
                "rows", "Rows to display", 10, 50, value=25, step=5, input_type="number"
            ),
            ui.action_button("clear", "Clear History"),
            ui.action_button("freeze", "Freeze Output"),
            ui.selectize(
                "colormap",
                "Colormap",
                options=[
                    "Viridis",
                    "Plasma",
                    "Inferno",
                    "Magma",
                    "Cividis",
                ],
                value="Viridis",
            ),
        ),
        ui.column(
            6,
            ui.h3("CPU Usage Data"),
            ui.table_output("table"),
        ),
    )
)

def server(input: Inputs, output: Outputs, session):
    cpu_data = reactive.Value(dict(time=[], cpu_percent=[]))
    plot = render.plot()
    table = render.table()

    @input.plot("plot")
    @reactive.event(cpu_data)
    def update_plot_handler(plot, cpu_data):
        update_plot(plot, cpu_data.get())

    @input.table("table")
    @reactive.event(cpu_data)
    def update_table_handler(table, cpu_data):
        df = pd.DataFrame(cpu_data.get())
        table(df.tail(input.rows()))

    @input.slider("samples")
    @reactive.event(cpu_data)
    def update_samples_handler(samples, cpu_data):
        df = pd.DataFrame(cpu_data.get())
        df = df.tail(samples)
        cpu_data(df.to_dict("records"))

    @input.slider("rows")
    @reactive.event(cpu_data)
    def update_rows_handler(rows, cpu_data):
        pass

    @input.action_button("clear")
    @reactive.event(cpu_data)
    def clear_handler(clear, cpu_data):
        cpu_data.set(dict(time=[], cpu_percent=[]))

    @input.action_button("freeze")
    @reactive.event(cpu_data)
    def freeze_handler(freeze, cpu_data):
        pass

    @input.selectize("colormap")
    @reactive.event(cpu_data)
    def colormap_handler(colormap, cpu_data):
        pass

    async def fetch_cpu_data():
        data = await get_cpu_data()
        cpu_data(data)

    session.loop.create_task(fetch_cpu_data())

app = App(app_ui, server)
```