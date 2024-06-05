  ```
import os
from pyaedt import Hfss

# Set non-graphical mode
non_graphical = False

# Create a new Hfss project
project_name = "MyProject"
project = Hfss(project_name, non_graphical, new_project=True)

# Insert a Maxwell 2D design
design_name = "MyDesign"
design = project.insert_design(design_name, "Maxwell 2D")

# Save the project
project.save_project()

# Create a rectangle
rectangle = design.modeler.create_rectangle(
    position=[0, 0],
    dimension=[10, 10],
    name="MyRectangle",
)

# Duplicate and unite the rectangle
rectangle_duplicate = rectangle.duplicate_and_unite(
    position=[10, 10],
    name="MyRectangleDuplicate",
)

# Create an air region
air_region = design.modeler.create_air_region(
    position=[0, 0],
    dimension=[10, 10],
    name="MyAirRegion",
)

# Assign windings to the sheets
winding_1 = design.modeler.create_winding(
    position=[0, 0],
    dimension=[10, 10],
    name="MyWinding1",
)
winding_2 = design.modeler.create_winding(
    position=[10, 10],
    dimension=[10, 10],
    name="MyWinding2",
)

# Assign a balloon to the air region
balloon = design.modeler.create_balloon(
    position=[0, 0],
    dimension=[10, 10],
    name="MyBalloon",
)

# Plot the model
design.plot(show=False)

# Create a transient setup
setup_name = "MySetup"
setup = design.create_setup(setup_name, "Transient")

# Create a rectangular plot
plot_name = "MyPlot"
plot = setup.create_rectangular_plot(plot_name, [0, 0], [10, 10])

# Solve the model
setup.analyze_nominal()

# Create output and plot it using PyVista
from pyvista import Plotter

plotter = Plotter(plot)
plotter.show()

# Generate the same plot outside AEDT
import matplotlib.pyplot as plt

plt.plot(plot.x, plot.y)
plt.show()

# Close AEDT
project.close_project()
```