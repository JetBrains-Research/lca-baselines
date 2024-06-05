 Here is a simplified version of the Bokeh application you requested. Please note that this code assumes you have a `load_population` function that returns population data and the necessary libraries are installed.

```python
from bokeh.io import output_file, show, save
from bokeh.layouts import row, column
from bokeh.models import ColumnDataSource, CategoricalColorMapper, SelectDateRange, Select
from bokeh.plotting import Figure, figure, bar, line
from bokeh.models.widgets import DataTable, TableColumn
from bokeh.application import Application
from bokeh.application.handlers import FunctionHandler
from bokeh.runtime import callback_when
from bokeh.util.browser import view
from bokeh.models.widgets import TextInput

# Assuming load_population returns a dictionary with keys 'year', 'location', 'male_known', 'male_predicted', 'female_known', 'female_predicted'
data = load_population()

source = ColumnDataSource(data)

# Population Pyramid
pyramid = figure(x_axis_type='category', plot_width=800, plot_height=400, title="Population Pyramid")
pyramid.vbar(x='location', top='male_known', bottom='male_predicted', width=0.9, color=['#FF0000', '#00FF00'], source=source)
pyramid.vbar(x='location', top='female_known', bottom='female_predicted', width=0.9, color=['#0000FF', '#FF00FF'], source=source)

# Line Chart
line_chart = figure(title="Known vs Predicted Population", x_axis_label='Year', y_axis_label='Population', plot_width=800, plot_height=400)
line_chart.line(x='year', y='male_known', source=source, legend_label="Male - Known", line_color='#FF0000', alpha=0.8)
line_chart.line(x='year', y='male_predicted', source=source, legend_label="Male - Predicted", line_color='#FFA500', alpha=0.8)
line_chart.line(x='year', y='female_known', source=source, legend_label="Female - Known", line_color='#0000FF', alpha=0.8)
line_chart.line(x='year', y='female_predicted', source=source, legend_label="Female - Predicted", line_color='#FF00FF', alpha=0.8)

year_select = SelectDateRange(title="Year", start=data['year'].min(), end=data['year'].max(), value=data['year'].max())
location_select = Select(title="Location", options=data['location'].unique())

def update_plots(attrname, old, new):
    data = load_population(year=new.value, location=location_select.value)
    source.data = data

callback_when(year_select, 'value', update_plots)

layout = column(pyramid, line_chart, row(year_select, location_select))

output_file("widget.html")
save(Application(title="Population Data Visualization", root=layout))
```

This code creates a Bokeh application with two plots: a population pyramid and a line chart. The application allows users to select a year and a location, and the plots update based on these selections. The application is served using Bokeh server and the layout is saved into an HTML file named "widget.html". The application continues running until it is manually stopped.