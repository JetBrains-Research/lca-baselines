  from bokeh.plotting import figure, output_file, show
from bokeh.models import ColumnDataSource, HoverTool, Select
from bokeh.layouts import column
from bokeh.server.server import Server

# Create a ColumnDataSource for the population data
source = ColumnDataSource(data=dict(
    year=[2010, 2015, 2020],
    location=['USA', 'Canada', 'Mexico'],
    male=[100, 120, 140],
    female=[120, 140, 160],
    known=[100, 120, 140],
    predicted=[120, 140, 160]
))

# Create a ColumnDataSource for the hover tool
hover = ColumnDataSource(data=dict(
    year=[],
    location=[],
    male=[],
    female=[],
    known=[],
    predicted=[]
))

# Create the population pyramid plot
pyramid = figure(title="Population Pyramid", x_axis_label="Year", y_axis_label="Population")
pyramid.vbar(x='year', top='male', bottom='female', width=0.5, color='#66c2a5', legend='Male')
pyramid.vbar(x='year', top='female', bottom='male', width=0.5, color='#fc8d62', legend='Female')
pyramid.legend.location = "top_left"
pyramid.legend.click_policy = "hide"

# Create the line chart plot
chart = figure(title="Population Trend", x_axis_label="Year", y_axis_label="Population")
chart.line(x='year', y='known', color='#66c2a5', legend='Known')
chart.line(x='year', y='predicted', color='#fc8d62', legend='Predicted')
chart.legend.location = "top_left"
chart.legend.click_policy = "hide"

# Create the layout
layout = column(pyramid, chart)

# Create the Bokeh server
server = Server(layout, port=5006)

# Create the HTML file
output_file("widget.html", title="Population Visualization")

# Run the Bokeh server
server.start()

# Keep the server running until manually stopped
while True:
    pass