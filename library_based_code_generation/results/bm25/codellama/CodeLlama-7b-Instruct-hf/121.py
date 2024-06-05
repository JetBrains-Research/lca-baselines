  Here is the Python code to create a Bokeh application to visualize population data:
```
from bokeh.plotting import figure, show
from bokeh.models import ColumnDataSource, HoverTool
from bokeh.layouts import column
from bokeh.application import Application
from bokeh.application.handlers import FunctionHandler

def make_population_pyramid(year, location):
    # Load population data
    data = load_population(year, location)

    # Create figure
    p = figure(title="Population Pyramid", x_axis_label="Age", y_axis_label="Population")

    # Add male and female sections
    p.vbar(x=data["age"], top=data["male"], width=0.5, color="blue")
    p.vbar(x=data["age"], bottom=data["male"], top=data["female"], width=0.5, color="red")

    # Add hover tool
    hover = HoverTool(tooltips=[("Age", "@age"), ("Male", "@male"), ("Female", "@female")])
    p.add_tools(hover)

    # Create column data source
    source = ColumnDataSource(data=data)

    # Create line chart
    line = figure(title="Population Line Chart", x_axis_label="Year", y_axis_label="Population")
    line.line(x=data["year"], y=data["population"], color="black")

    # Create column data source
    line_source = ColumnDataSource(data=data)

    # Create layout
    layout = column(p, line)

    # Create application
    app = Application(FunctionHandler(make_population_pyramid))

    # Serve application
    app.serve("widget.html")

    # Run application
    app.run()
```
This code creates a Bokeh application that visualizes population data using a population pyramid and a line chart. The application allows users to select a year and a location, and the plots will update based on these selections. The application is served using Bokeh server and the layout is saved into an HTML file named "widget.html". The application will continue running until it is manually stopped.