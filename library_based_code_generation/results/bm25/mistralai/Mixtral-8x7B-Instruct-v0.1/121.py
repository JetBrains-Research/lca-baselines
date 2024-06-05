 from bokeh.io import curdoc
from bokeh.layouts import row
from bokeh.models import ColumnDataSource, Slider, Select
from bokeh.plotting import figure
import bokeh.palettes as palettes

from population_data import load\_population

population\_data = load\_population()

source = ColumnDataSource(data=dict(year=[2010], location=['USA'], male=[], female=[], known=[], predicted=[]))

def update\_data(attrname, old, new):
year = int(callback.value)
location = select.value

filtered\_data = population\_data[(population\_data['year'] == year) & (population\_data['location'] == location)]

source.data = dict(year=[year], location=[location], male=filtered\_data['male'].values, female=filtered\_data['female'].values, known=filtered\_data['known'].values, predicted=filtered\_data['predicted'].values)

year\_slider = Slider(start=min(population\_data['year']), end=max(population\_data['year']), value=min(population\_data['year']), step=1, title="Year", callback=update\_data)

location\_select = Select(title="Location", value="USA", options=sorted(population\_data['location'].unique()))

callback = Slider(start=min(population\_data['year']), end=max(population\_data['year']), value=min(population\_data['year']), step=1, title="Year", callback=update\_data)

p1 = figure(x_axis_type="category", title="Population Pyramid", width=800, height=400)
p1.vbar\_stack(stackers=['male', 'female'], x='year', width=0.5, source=source, alpha=0.8, line\_color="white", legend_label=["Male", "Female"])

p2 = figure(title="Population Over Time", width=800, height=400)
p2.line(x='year', y='known', color=palettes.Set3[0], alpha=0.8, legend_label="Known", source=source)
p2.line(x='year', y='predicted', color=palettes.Set3[1], alpha=0.8, legend_label="Predicted", source=source)

curdoc().add\_root(row(year\_slider, location\_select, p1, p2))
curdoc().title = "Population Visualization"

if __name__ == "__main__":
curdoc().application\_context.notebook\_handler = None
curdoc().application\_context.server\_context.notebook\_mode = False
curdoc().application\_context.server\_context.allow\_websocket\_origin = ["*"]
curdoc().application\_context.server\_context.webSocket\_origin = ["*"]
curdoc().application\_context.server\_context.port = 5006
curdoc().application\_context.server\_context.destination = "localhost"
curdoc().application\_context.server\_context.url\_prefix = ""
curdoc().application\_context.server\_context.show = True

from bokeh.application import Application
from bokeh.application.handlers.function import FunctionHandler
from bokeh.util.browser import view

handler = FunctionHandler(screen_app)
app = Application(handler)

view(app)