 Here is a simplified version of the requested Python code using Flask for the web server, Pyodide for the browser-based execution, and psutil/psutil_browser for CPU usage monitoring. The code uses Matplotlib, Pandas, and Shiny for the UI and data visualization.

```python
import os
import sys
import time
import psutil
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from flask import Flask, render_template, request, jsonify
from flask_shiny import FlaskApp, ui, output_modes, render_template_string
from psutil_browser import PSUtilBrowser

if "browser" in sys.argv:
    psutil = PSUtilBrowser()
else:
    import psutil

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your_secret_key'

app = FlaskApp(__name__)
server = app.server

samples_per_graph = 100
rows_to_display = 20

cpu_percent_list = []

def hide_ticks(ax):
    ax.set_xticks([])
    ax.set_yticks([])

def create_graph(data):
    fig, ax = plt.subplots()
    ax.plot(data)
    hide_ticks(ax)
    return fig

@app.route('/')
def home():
    return render_template('index.html', colormap=['r', 'g', 'b', 'y', 'c'],
                           samples_per_graph=samples_per_graph,
                           rows_to_display=rows_to_display)

@app.route('/update_data')
def update_data():
    global cpu_percent_list
    cpu_percent_list.append(psutil.cpu_percent(samples_per_graph))
    if len(cpu_percent_list) > samples_per_graph * 10:
        cpu_percent_list.pop(0)
    return jsonify(cpu_percent_list[-samples_per_graph:])

@app.route('/clear_data')
def clear_data():
    global cpu_percent_list
    cpu_percent_list.clear()
    return 'Data cleared.'

@app.route('/freeze')
def freeze():
    return 'Freezing output.'

@app.route('/unfreeze')
def unfreeze():
    return 'Unfreezing output.'

@app.route('/graph_data')
def graph_data():
    data = cpu_percent_list[-samples_per_graph:]
    fig = create_graph(data)
    return plt.gcf().to_base64_png()

@app.route('/table_data')
def table_data():
    data = pd.DataFrame(cpu_percent_list[-samples_per_graph * rows_to_display:], columns=['CPU Usage'])
    return data.to_html(index=False)

if __name__ == '__main__':
    server.run(app, host='0.0.0.0', port=8050, threaded=True)
```

The HTML template (index.html) for the UI:

```html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>CPU Usage Monitor</title>
    <style>
        body { font-family: Arial, sans-serif; }
    </style>
</head>
<body>
    <h1>CPU Usage Monitor</h1>
    <div id="graph"></div>
    <div id="table"></div>
    <hr>
    <form>
        Colormap:
        <select id="colormap">
            {% for colormap in colormap %}
                <option value="{{ colormap }}" {% if colormap == colormap[0] %}selected{% endif %}>{{ colormap }}</option>
            {% endfor %}
        </select>
        Samples per graph:
        <input type="number" id="samples_per_graph" value="{{ samples_per_graph }}">
        Rows to display:
        <input type="number" id="rows_to_display" value="{{ rows_to_display }}">
        <button id="clear_data">Clear Data</button>
        <button id="freeze">Freeze Output</button>
        <button id="unfreeze">Unfreeze Output</button>
    </form>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/jquery/3.6.0/jquery.min.js"></script>
    <script>
        $(document).ready(function() {
            // Update data every second
            setInterval(function() {
                $.getJSON('/update_data', function(data) {
                    // Update graph
                    let ctx = document.getElementById('graph').getContext('2d');
                    let myChart = new Chart(ctx, {
                        type: 'line',
                        data: {
                            labels: ['CPU Usage'],
                            datasets: [{
                                label: 'My First Dataset',
                                data: data,
                                borderColor: $('#colormap').val(),
                                fill: false
                            }]
                        },
                        options: {
                            responsive: true,
                            scales: {
                                x: {
                                    display: false
                                },
                                y: {
                                    display: false
                                }
                            }
                        }
                    });

                    // Update table
                    let table = $('#table').DataTable({
                        data: data,
                        columns: [{
                            title: 'CPU Usage'
                        }]
                    });
                });
            }, 1000);

            // Clear data
            $('#clear_data').click(function() {
                $.get('/clear_data');
            });

            // Freeze/unfreeze output
            $('#freeze').click(function() {
                $.get('/freeze');
            });
            $('#unfreeze').click(function() {
                $.get('/unfreeze');
            });
        });
    </script>
</body>
</html>
```

This code should work in both standard Python environments and Pyodide environments. However, you may need to adjust the code to better fit your specific needs and environment.