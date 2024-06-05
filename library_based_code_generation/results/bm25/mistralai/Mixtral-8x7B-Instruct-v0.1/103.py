 ```python
import PyAEDT
from pyaedt import Maxwell, Hfss, Icepak
from pyaedt.generic.general_methods import get_app_path
import pyvista as pv
import numpy as np

# Start with non-graphical mode
PyAEDT.set_mode("no_gui")

# Create Maxwell 2D project
maxwell = Maxwell()
maxwell.new_project("Maxwell_2D_Project", "Maxwell2D")
maxwell.set_active_project("Maxwell_2D_Project")
maxwell.set_active_design("Maxwell_2D_Project")

# Create a rectangle and duplicate it
rectangle = maxwell.box(name="Rectangle", layer="Layer1", x=1, y=1, z=0.01)
duplicated_rectangle = duplicate_and_unite(rectangle, "duplicated_rectangle", 1, 1, 0, 0, 0, 180)

# Create an air region
air_region = create_air_region(maxwell, "AirRegion", 10, 10, 0.01)

# Assign windings to the sheets and a balloon to the air region
maxwell.assign_current("Rectangle", "Layer1", 1, 1, 0, 0, 0, 1, 1, 1)
maxwell.assign_current("duplicated_rectangle", "Layer1", 1, 1, 0, 0, 0, 1, 1, 1)
maxwell.assign_balloon(air_region, "AirRegion", 1, 1, 0, 0, 0, 1)

# Plot the model
maxwell.plot_2d_chart(rectangle)

# Create a transient setup
maxwell.new_transient_study("TransientStudy", "TransientStudy", "1ns", "1ns", "1ns")

# Create a rectangular plot
rectangular_plot(maxwell, "RectangularPlot", 0, 0, 10, 10, 0, 0, 1, 1)

# Solve the model
maxwell.solve()

# Create output and plot it using PyVista
x, y, z = maxwell.get_3d_data("RectangularPlot", "ElectricField", "magnitude")
plotter = pv.Plotter()
plotter.add_mesh(pv.PolyData(np.column_stack((x, y, z))), style="wireframe")
plotter.show()

# Generate the same plot outside AEDT
output_file = get_app_path("Maxwell_2D_Project", "TransientStudy", "RectangularPlot", "ElectricField", "magnitude")
check_and_download_file(maxwell, output_file)
pv.read(output_file).plot()

# Close AEDT
maxwell.close_project()
```