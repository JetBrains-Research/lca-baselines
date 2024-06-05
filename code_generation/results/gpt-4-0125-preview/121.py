```python
from bokeh.models import ColumnDataSource, Select
from bokeh.layouts import column, row
from bokeh.plotting import figure, curdoc
from bokeh.io import output_file, save
import pandas as pd

# Sample data: Replace with your actual data source
data = {
    'Year': [2000, 2005, 2010, 2015, 2020, 2025, 2030],
    'Location': ['Location1', 'Location1', 'Location1', 'Location1', 'Location1', 'Location1', 'Location1'],
    'Male': [300, 320, 340, 360, 380, 400, 420],
    'Female': [280, 300, 320, 340, 360, 380, 400],
    'Total': [580, 620, 660, 700, 740, 780, 820],
    'Predicted': [False, False, False, True, True, True, True]
}
df = pd.DataFrame(data)

# Initial values
initial_year = 2000
initial_location = 'Location1'

# Data source for plots
source = ColumnDataSource(data=df[(df['Year'] == initial_year) & (df['Location'] == initial_location)])

# Population pyramid
p_pyramid = figure(y_range=(-500, 500), title="Population Pyramid")
p_pyramid.hbar(y='Male', right=0, height=0.4, left='Male', source=source, color="blue", legend_label="Male")
p_pyramid.hbar(y='Female', right='Female', height=0.4, source=source, color="pink", legend_label="Female")

# Line chart for population
p_line = figure(title="Population Over Years")
p_line.line(x='Year', y='Total', source=source, color="green", legend_label="Total Population")
p_line.circle(x='Year', y='Total', source=source, color="green", size=8)
p_line.line(x='Year', y='Total', source=source, color="red", legend_label="Predicted Population", line_dash="dashed")
p_line.circle(x='Year', y='Total', source=source, color="red", size=8)

# Widgets
year_select = Select(title="Year", value=str(initial_year), options=[str(year) for year in df['Year'].unique()])
location_select = Select(title="Location", value=initial_location, options=df['Location'].unique().tolist())

# Callbacks
def update(attr, old, new):
    year = int(year_select.value)
    location = location_select.value
    new_data = df[(df['Year'] == year) & (df['Location'] == location)]
    source.data = ColumnDataSource.from_df(new_data)

year_select.on_change('value', update)
location_select.on_change('value', update)

# Layout
layout = column(row(year_select, location_select), p_pyramid, p_line)

# Output
output_file("widget.html")
save(layout)

# Add to document
curdoc().add_root(layout)
curdoc().title = "Population Visualization"

# Note: To run this, save the code to a file (e.g., app.py) and use the Bokeh server to run it:
# bokeh serve --show app.py
# The application will continue running until manually stopped.
```