  ```
import os
from pyaedt import Maxwell2d

# Set non-graphical mode
non_graphical = True

# Create a new Maxwell 2D design
maxwell = Maxwell2d(projectname="MyMaxwell2DDesign", designname="MyDesign",
                    solution_type="Transient", non_graphical=non_graphical)

# Insert a rectangle and duplicate it
maxwell.modeler.create_rectangle(position=[0, 0], dimensions=[10, 10], name="Rectangle1")
maxwell.modeler.duplicate_object("Rectangle1", "Rectangle2")

# Create an air region
maxwell.modeler.create_air_region(name="AirRegion")

# Assign windings to the sheets
maxwell.assign_winding_to_sheet("Rectangle1", "Winding1")
maxwell.assign_winding_to_sheet("Rectangle2", "Winding2")

# Assign a balloon to the air region
maxwell.assign_balloon_to_air_region("AirRegion", "Balloon1")

# Plot the model
maxwell.plot(show=False)

# Create a transient setup
maxwell.create_setup(setupname="TransientSetup", solutype="Transient")

# Create a rectangular plot
maxwell.create_plot(plotname="RectangularPlot", plot_type="RectangularPlot",
                    x_axis_variable="X", y_axis_variable="Y",
                    x_axis_increment=1, y_axis_increment=1,
                    x_axis_start=0, y_axis_start=0,
                    x_axis_end=10, y_axis_end=10)

# Solve the model
maxwell.analyze_setup("TransientSetup")

# Create output and plot it using PyVista
maxwell.post.create_output(output_type="VTK", output_name="MyOutput")
maxwell.post.plot_output(output_name="MyOutput", plot_type="RectangularPlot",
                         x_axis_variable="X", y_axis_variable="Y",
                         x_axis_increment=1, y_axis_increment=1,
                         x_axis_start=0, y_axis_start=0,
                         x_axis_end=10, y_axis_end=10)

# Generate the same plot outside AEDT
import pyvista as pv

pv.plot(maxwell.post.get_output(output_name="MyOutput", plot_type="RectangularPlot"))

# Close AEDT
maxwell.close_project()
```