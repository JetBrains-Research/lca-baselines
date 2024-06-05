from bokeh.plotting import figure, curdoc
from bokeh.layouts import column, row
from bokeh.models import ColumnDataSource, Select
from bokeh.io import output_file, show

# Create population pyramid plot
pyramid = figure(title="Population Pyramid", x_range=(0, 100), y_range=(0, 100), plot_height=400, plot_width=400)
pyramid.vbar(x='age', top='male_population', width=0.5, source=source, color='blue', legend_label='Male')
pyramid.vbar(x='age', top='female_population', width=0.5, source=source, color='pink', legend_label='Female')

# Create line chart plot
line_chart = figure(title="Population Trend", x_axis_label='Year', y_axis_label='Population', plot_height=400, plot_width=800)
line_chart.line(x='year', y='known_population', source=source, line_width=2, color='blue', legend_label='Known')
line_chart.line(x='year', y='predicted_population', source=source, line_width=2, color='red', legend_label='Predicted')

# Create Select widgets for year and location
year_select = Select(title="Select Year:", options=['2020', '2021', '2022'], value='2020')
location_select = Select(title="Select Location:", options=['City A', 'City B', 'City C'], value='City A')

# Define callback function to update plots based on user selections
def update_data(attrname, old, new):
    # Update data based on selected year and location
    selected_year = year_select.value
    selected_location = location_select.value
    new_data = get_updated_data(selected_year, selected_location)
    source.data = new_data

# Add callback to Select widgets
year_select.on_change('value', update_data)
location_select.on_change('value', update_data)

# Create ColumnDataSource
source = ColumnDataSource(data=get_initial_data())

# Create layout
layout = column(row(year_select, location_select), row(pyramid, line_chart))

# Add layout to current document
curdoc().add_root(layout)

# Save layout to HTML file
output_file("widget.html")

# Run the Bokeh application
curdoc().title = "Population Visualization"
curdoc().add_root(layout)
curdoc().title = "Population Visualization"
curdoc().add_root(layout)
curdoc().title = "Population Visualization"
curdoc().add_root(layout)