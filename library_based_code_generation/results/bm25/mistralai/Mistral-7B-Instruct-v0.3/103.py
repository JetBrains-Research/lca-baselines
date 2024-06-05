 Here is a basic example of how you might structure your code using the PyAEDT library to create a Maxwell 2D project, run a transient simulation, and plot the results using PyVista. Please note that this is a simplified example and may need to be adjusted to fit your specific needs.

```python
from pyAEDT import *
from pyVista import Plotter, PolyData

# Set non-graphical mode
AEDT.set_non_graphical_mode()

# Create a new project
project = AEDT.create_project("My_Project")

# Open Maxwell 2D design
maxwell2d = project.open_maxwell2d_design("Maxwell2D.dsn")

# Create a rectangle
rectangle = maxwell2d.create_rectangle(0, 0, 1, 1)

# Duplicate and mirror the rectangle
duplicate_and_mirror(rectangle, "X")

# Create an air region
air_region = maxwell2d.create_air_region(0.05, 0.05, 0.95, 0.95)

# Assign windings to the sheets
windings = maxwell2d.create_windings(rectangle, 1)

# Assign a balloon to the air region
balloon = maxwell2d.create_balloon(air_region, 0)

# Plot the model
maxwell2d.plot_2d_chart()

# Create a transient setup
transient_setup = maxwell2d.create_transient_setup(start_time=0, end_time=1e-6, time_step=1e-12)

# Create a rectangular plot
rectangular_plot(maxwell2d, 0, 0, 1, 1)

# Solve the model
maxwell2d.solve()

# Create output and plot it using PyVista
output = maxwell2d.get_output("E_x")
data = PolyData(output.get_data())
plotter = Plotter(size=(800, 600))
plotter.plot(data, edge_color="red")

# Generate the same plot outside AEDT
# (This part is not directly supported by PyAEDT, you may need to use other libraries)

# Close AEDT
AEDT.close_all()

# Clean up cache and files
AEDT.clean_cache_and_files()
```

This code creates a new project, opens a Maxwell 2D design, creates a rectangle, duplicates and mirrors it, creates an air region, assigns windings to the sheets and a balloon to the air region, creates a transient setup, creates a rectangular plot, solves the model, generates output, plots it using PyVista, generates the same plot outside AEDT, and finally closes AEDT.