 from bokeh.io import curdoc
from bokeh.layouts import row
from bokeh.models import ColumnDataSource, Slider, Select
from bokeh.plotting import figure

import pandas as pd

# Assuming you have a dataframe `population_data` with columns: 'Year', 'Location', 'Gender', 'Population', 'Type'
population_data = pd.read_csv('population_data.csv')

cds = ColumnDataSource(population_data)

p1 = figure(x_range=(-120, 120), y_range=(0, max(population_data['Population']) * 1.1), height=300, width=800, title="Population Pyramid", toolbar_location=None)
p1.vbar(x='Gender', top='Population', width=0.9, source=cds, bottom=0, alpha=0.9, color=['#6495ED', '#FF69B4'])
p1.xaxis.major_label_orientation = pi/4
p1.xaxis.axis_label = 'Gender'
p1.yaxis.axis_label = 'Population'

p2 = figure(height=300, width=800, title="Population Line Chart", toolbar_location=None)
p2.line(x='Year', y='Population', line_width=2, alpha=0.8, color='#1F77B4', source=cds, legend_label='Known')
p2.line(x='Year', y='Population', line_width=2, alpha=0.8, color='#FFC107', source=cds, legend_label='Predicted')
p2.legend.location = 'top_left'

def update_data(attrname, old, new):
    year = year_slider.value
    location = location_select.value
    filtered_data = population_data[(population_data['Year'] == year) & (population_data['Location'] == location)]
    cds.data = ColumnDataSource(filtered_data).data

year_slider = Slider(start=min(population_data['Year']), end=max(population_data['Year']), value=min(population_data['Year']), step=1, title="Year")
year_slider.js_on_change('value', update_data)

location_select = Select(title="Location", value=population_data['Location'].unique()[0], options=population_data['Location'].unique())
location_select.js_on_change('value', update_data)

curdoc().add_root(row(year_slider, location_select, p1, p2))
curdoc().title = "Population Data Visualization"

export_html(curdoc(), "widget.html")
curdoc().validate()
curdoc().interrupt_callbacks()
curdoc().bind_to_jsevent(WindowEvent('resize'), 'reset')
curdoc()