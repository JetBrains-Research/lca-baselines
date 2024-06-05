```python
from bokeh.io import curdoc
from bokeh.layouts import column, row
from bokeh.models import ColumnDataSource, Select
from bokeh.plotting import figure
from bokeh.server.server import Server

# Create population pyramid plot
pyramid = figure(title="Population Pyramid", x_range=(-1000, 1000), y_range=(0, 100), plot_height=400, plot_width=400)
pyramid.hbar(y='age', right='male', height=0.5, source=source, color='blue', legend_label='Male')
pyramid.hbar(y='age', right='female', height=0.5, source=source, color='pink', legend_label='Female')

# Create line chart plot
line_chart = figure(title="Population Trend", x_axis_label='Year', y_axis_label='Population', plot_height=400, plot_width=800)
line_chart.line(x='year', y='population', source=source, color='blue', legend_label='Known')
line_chart.line(x='year', y='predicted_population', source=source, color='red', legend_label='Predicted')

# Create Select widgets
year_select = Select(title="Year:", options=['2020', '2021', '2022'], value='2020')
location_select = Select(title="Location:", options=['City', 'Town', 'Village'], value='City')

# Define callback function
def update_data(attrname, old, new):
    # Update data based on user selections
    year = int(year_select.value)
    location = location_select.value
    new_data = get_updated_data(year, location)
    source.data = new_data

# Add callback to Select widgets
year_select.on_change('value', update_data)
location_select.on_change('value', update_data)

# Create layout
layout = column(year_select, location_select, row(pyramid, line_chart))

# Save layout into HTML file
curdoc().add_root(layout)
curdoc().title = "Population Data Visualization"
curdoc().add_periodic_callback(update_data, 1000)

# Serve the application
server = Server({'/': layout}, num_procs=1)
server.start()
server.io_loop.add_callback(server.show, "/")
server.io_loop.start()
```