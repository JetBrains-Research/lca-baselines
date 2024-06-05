```python
import os
from pyaedt import HFSS3DLayout

temp_dir = os.path.join(os.getcwd(), "temp_dir")
if not os.path.exists(temp_dir):
    os.makedirs(temp_dir)

hfss = HFSS3DLayout()
hfss.check_and_download_file("dcir_example_project", temp_dir)

hfss.create_port_between_pin_and_layer("VRM_positive", "VRM_negative")
hfss.create_voltage_source_on_pin("VRM_positive", "VRM_negative")

hfss.create_port_between_pin_and_layer("sink_positive", "sink_negative")
hfss.create_current_source_on_pin("sink_positive", "sink_negative")

hfss.add_siwave_dcir_analysis()
hfss.save_project()
hfss.close_project()

hfss.launch_aedt()
hfss.import_edb(temp_dir)
hfss.analyze_dcir()

loop_resistance = hfss.get_dcir_element_data_loop_resistance()
current_source = hfss.get_dcir_element_data_current_source()
via_info = hfss.get_dcir_element_data_via()
voltage = hfss.get_dcir_solution_data()

print("Loop Resistance:", loop_resistance)
print("Current Source:", current_source)
print("Via Information:", via_info)
print("Voltage:", voltage)

hfss.close_project()
hfss.release_desktop()
```