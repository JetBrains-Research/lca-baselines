```python
from bokeh.io import curdoc
from bokeh.layouts import column, row
from bokeh.models import ColumnDataSource, Select
from bokeh.plotting import figure
from bokeh.server.server import Server
from bokeh.application import Application
from bokeh.application.handlers.function import FunctionHandler

def modify_doc(doc):
    # Sample data
    data = {
        'Year': [2000, 2005, 2010, 2015, 2020, 2025, 2030],
        'Known_Population': [6e9, 6.5e9, 7e9, 7.3e9, 7.8e9, None, None],
        'Predicted_Population': [None, None, None, None, 7.8e9, 8e9, 8.5e9],
        'Male': [3e9, 3.25e9, 3.5e9, 3.65e9, 3.9e9, 4e9, 4.25e9],
        'Female': [3e9, 3.25e9, 3.5e9, 3.65e9, 3.9e9, 4e9, 4.25e9]
    }

    source = ColumnDataSource(data=data)

    # Widgets
    year_select = Select(title="Year", value="2000", options=[str(year) for year in data['Year']])
    location_select = Select(title="Location", value="Global", options=["Global"])  # Placeholder for location options

    # Population pyramid
    pyramid_male = figure(title="Population Pyramid", x_axis_label="Population", y_axis_label="Age Group", tools="")
    pyramid_male.hbar(y=[1, 2, 3], right=data['Male'][:3], height=0.9, color="blue", legend_label="Male")

    pyramid_female = figure(title="Population Pyramid", x_axis_label="Population", y_axis_label="Age Group", tools="")
    pyramid_female.hbar(y=[1, 2, 3], right=data['Female'][:3], height=0.9, color="pink", legend_label="Female")

    # Line chart for known and predicted population
    line_chart = figure(title="Population Over Years", x_axis_label="Year", y_axis_label="Population", tools="")
    line_chart.line(x='Year', y='Known_Population', source=source, legend_label="Known Population", color="green")
    line_chart.line(x='Year', y='Predicted_Population', source=source, legend_label="Predicted Population", color="red")

    # Callbacks to update plots
    def update(attr, old, new):
        year = int(year_select.value)
        # Placeholder for updating data based on year and location
        # This is where you would filter or update your data source based on the selected year and location
        # For demonstration, this does nothing substantial but represents where such logic would go
        print(f"Year: {year}, Location: {location_select.value}")

    year_select.on_change('value', update)
    location_select.on_change('value', update)

    # Layout
    layout = column(row(year_select, location_select), row(pyramid_male, pyramid_female), line_chart)
    doc.add_root(layout)

# Bokeh application
app = Application(FunctionHandler(modify_doc))

# Save the layout into an HTML file
from bokeh.io import output_file, save
output_file("widget.html")
save(app.create_document())

# Running the server
def bk_worker():
    server = Server({'/': app}, io_loop=IOLoop.current())
    server.start()
    server.io_loop.start()

from tornado.ioloop import IOLoop
from threading import Thread

Thread(target=bk_worker).start()
```