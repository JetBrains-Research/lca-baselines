```python
from pyaedt import Edb
from pyaedt.generic.filesystem import Scratch
import os

# Create a temporary directory and set up EDB
scratch = Scratch("siwave_dcir_analysis")
edb_path = os.path.join(scratch.path, "example_board.edb")
if not os.path.exists(edb_path):
    Edb.test_54b_open_and_load_a_polyline(edb_path, edbversion="2021.2")

# Initialize EDB object
edb = Edb(edbpath=edb_path, edbversion="2021.2")

# Create pin groups for VRM and sink component
vrm_pos_pins = ["U2A5", "U2A6"]
vrm_neg_pins = ["U2B5", "U2B6"]
sink_pos_pins = ["U10A1", "U10A2"]
sink_neg_pins = ["U10B1", "U10B2"]

edb.core_components.pin_groups.create_pin_group_from_pins("VRM_POS", vrm_pos_pins)
edb.core_components.pin_groups.create_pin_group_from_pins("VRM_NEG", vrm_neg_pins)
edb.core_components.pin_groups.create_pin_group_from_pins("SINK_POS", sink_pos_pins)
edb.core_components.pin_groups.create_pin_group_from_pins("SINK_NEG", sink_neg_pins)

# Create voltage and current sources
edb.core_siwave.create_voltage_source_on_pin("V1", "VRM_POS", "VRM_NEG", 3.3)
edb.core_siwave.create_current_source_on_pin("I1", "SINK_POS", "SINK_NEG", 0.5)

# Save and close EDB
edb.save_edb()
edb.close_edb()

# Launch AEDT, import EDB, and analyze DCIR
from pyaedt import Hfss3dLayout

aedt_app = Hfss3dLayout(specified_version="2021.2")
aedt_app.import_edb(edb_path)
aedt_app.create_setup(setupname="DCIRSetup", setuptype="DCIR")
aedt_app.analyze_setup("DCIRSetup")

# Retrieve and print DCIR analysis results
loop_resistance = aedt_app.get_dcir_element_data_loop_resistance("I1")
current_source_info = aedt_app.get_dcir_element_data_current_source("I1")
via_info = aedt_app.get_dcir_element_data_via()
voltage = aedt_app.get_dcir_solution_data("V1")

print("Loop Resistance:", loop_resistance)
print("Current Source Info:", current_source_info)
print("Via Information:", via_info)
print("Voltage:", voltage)

# Close the AEDT project and release the desktop
aedt_app.close_project(aedt_app.project_name)
aedt_app.release_desktop()
```