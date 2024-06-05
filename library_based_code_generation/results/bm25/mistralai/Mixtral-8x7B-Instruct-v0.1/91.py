 from psutil.pseudo import cpu_percent
import psutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from functools import partial
from shiny import App, Inputs, Outputs, Session, reactive, render, ui

def fake_cpu_percent():
return np.random.rand() * 100

psutil_cpu_percent = psutil.cpu_percent if not in_pyodide() else fake_cpu_percent

def in_pyodide():
return "pyodide" in sys.implementation.name

def hide_ticks(ax):
ax.set_xticks([])
ax.set_yticks([])

def hist_data(n_samples):
return list(reactive({range(n_samples)}, session=Session.get_session())())

app_ui = ui.page_fluid(
ui.tags$h1("CPU Usage Monitoring Application"),
ui.row(
ui.column(
6,
ui.input_slider("n_samples", "Samples per graph", 10, 100, 50),
ui.input_slider("n_rows", "Rows to display", 10, 50, 25),
ui.input_selectize("colormap", "Colormap", options=["viridis", "plasma", "inferno"]),
ui.action_button("clear", "Clear history"),
ui.action_button("freeze", "Freeze output"),
ui.action_button("hide_ticks", "Hide ticks"),
),
ui.column(
6,
ui.output_graph("graph"),
ui.output_table("table"),
)
),
style="font-family:sans-serif;font-size:14px;"
)

def server(input, output, session):
@output
@render.plot
def graph():
n_samples = input.n_samples()
data = hist_data(n_samples)
fig, ax = plt.subplots()
ax.plot(data, color=session.input.colormap())
hide_ticks(ax)
return fig

@output
@render.table
def table():
n_rows = input.n_rows()
df = to_pandas(psutil_cpu_percent(), n_rows)
return DataTable(df)

@reactive.Effect(session)
def update_data():
while should_continue():
yield
n_samples = input.n_samples()
data.append(psutil_cpu_percent())
if len(data) > n_samples:
data.pop(0)

@reactive.Effect(session)
def update_table():
while should_continue():
yield
n_rows = input.n_rows()
df = to_pandas(psutil_cpu_percent(), n_rows)
output.table.set_datatable(df)

@reactive.Effect(session)
def update_graph():
while should_continue():
yield
n_samples = input.n_samples()
data = hist_data(n_samples)
output.graph.set_figure(fig)

@input.clear.validate
def validate_clear():
return not session.input.freeze()

@input.clear.react
def react_clear():
data.clear()

@input.freeze.validate
def validate_freeze():
return not session.input.freeze()

@input.freeze.react
def react_freeze():
session.input.freeze.set_not_deferred()

@input.hide_ticks.validate
def validate_hide_ticks():
return not session.input.freeze()

@input.hide_ticks.react
def react_hide_ticks():
fig = plt.gcf()
hide_ticks(fig.get_axes()[0])
fig.canvas.draw()

if __name__ == "__main__":
app = App(app_ui, server)
app.run(host="0.0.0.0", port=8080, debug=True)
else:
app.deploy_to_shinyapps()