 Here is a simplified version of the requested Python code using Flask for the web server, PyShiny for the UI, Pyodide for the browser execution, psutil for CPU usage, matplotlib for plotting, pandas for tabular data, and seaborn for colormaps. Please note that this is a basic implementation and might need adjustments to fit your specific needs.

```python
import os
import time
import psutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from flask import Flask, render_template, request, send_file
from pyshiny import ui, server
from pyshiny.pyodide import PyodideApp

def fake_psutil():
    cpu_percent = 0.0
    cpu_times_per_cpu = np.zeros((1, 3))

    def update():
        nonlocal cpu_percent
        cpu_percent += 0.1

    return cpu_percent, cpu_times_per_cpu, update

def hide_ticks(ax):
    ax.set_xticks([])
    ax.set_yticks([])

app = Flask(__name__)

if "browser" in locals():
    app = PyodideApp(app)

server = app.server

cpu_data = []
colormap = "viridis"
samples = 100
rows = 10
freeze = False

def sample_cpu():
    global cpu_data
    cpu_percent, _, _ = psutil.cpu_percent(interval=1) if not freeze else fake_psutil()
    cpu_data.append([time.time(), cpu_percent])
    if len(cpu_data) > samples:
        cpu_data.pop(0)

def plot_cpu():
    df = pd.DataFrame(cpu_data, columns=["Time", "CPU Percent"])
    fig, ax = plt.subplots()
    sns.lineplot(x="Time", y="CPU Percent", data=df, color=colormap)
    hide_ticks(ax)
    return fig

def table_cpu():
    df = pd.DataFrame(cpu_data, columns=["Time", "CPU Percent"])
    return df.head(rows)

@app.route("/")
def home():
    return render_template("index.html", plot=plot_cpu(), table=table_cpu())

@app.route("/update", methods=["POST"])
def update():
    global samples, rows, colormap, freeze
    samples = int(request.form["samples"])
    rows = int(request.form["rows"])
    colormap = request.form["colormap"]
    freeze = request.form["freeze"] == "on"
    return home()

if __name__ == "__main__":
    app.run(debug=True)
```

The HTML template (index.html) should be created separately and include the necessary Shiny and Matplotlib code to render the plot and table.

```html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>CPU Usage Monitor</title>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/shiny/1.7.1/shiny.min.js"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/matplotlib/3.5.1/embed.js"></script>
</head>
<body>
    <div id="plot" style="width: 800px; height: 600px;"></div>
    <div id="table"></div>
    <script>
        // Initialize Shiny and Matplotlib
        // ...

        // Render plot and table
        // ...

        // Update plot and table on user interactions
        // ...
    </script>
</body>
</html>
```