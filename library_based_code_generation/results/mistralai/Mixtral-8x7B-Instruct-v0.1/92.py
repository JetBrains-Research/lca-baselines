 ```python
import PyAEDT
from pyaedt import HFSS
from pyaedt.generic.general_functions import download_file
from pyaedt.generic.json_ import read_json
import os

PyAEDT.set_log_level("info")
PyAEDT.start_aedt(mode="non_graphical")

HFSS.download_3d_component("antennas", "HornAntenna_v01")
HFSS.new_project("Antenna_Array_Example", "HFSS")

array_def = read_json("array_definition.json")
components_path = {"HornAntenna_v01": "path/to/HornAntenna_v01.iam"}

for comp_name, comp_path in components_path.items():
    HFSS.desktop.open_3d_component(comp_path)
    HFSS.active_design.insert_component(comp_name)

HFSS.desktop.set_active_design("Antenna_Array_Example")
HFSS.active_design.analyze()

far_field_data = HFSS.active_design.get_far_field_data()

HFSS.active_design.post_process.contour_plot(far_field_data)
HFSS.active_design.post_process.cut_plot(far_field_data)
HFSS.active_design.post_process.polar_plot(far_field_data)

import matplotlib.pyplot as plt
import pyvista as pv

fig, ax = plt.subplots()
HFSS.active_design.post_process.polar_plot_matplotlib(far_field_data, ax)
plt.show()

pvd_file = HFSS.active_design.post_process.polar_plot_pyvista(far_field_data)
plotter = pv.Plotter()
plotter.open(pvd_file)
plotter.show()

HFSS.desktop.exit_project()
PyAEDT.stop_aedt()
```