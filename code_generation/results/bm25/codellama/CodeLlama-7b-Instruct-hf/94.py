  ```
import pyaedt

# Create a new EDB project
edb = pyaedt.Edb(edb_path)

# Configure EDB for DCIR analysis
edb.create_port_between_pin_and_layer(
    "VRM_Positive", "VRM_Negative", "VRM_Positive", "VRM_Negative"
)
edb.create_pin_groups(["VRM_Positive", "VRM_Negative"])
edb.create_voltage_source_on_pin("VRM_Positive", "VRM_Negative")

edb.create_port_between_pin_and_layer(
    "Sink_Positive", "Sink_Negative", "Sink_Positive", "Sink_Negative"
)
edb.create_pin_groups(["Sink_Positive", "Sink_Negative"])
edb.create_current_source_on_pin("Sink_Positive", "Sink_Negative")

# Add a SIwave DCIR analysis
edb.add_analysis("SIwaveDCIR")

# Save and close EDB
edb.save_project()
edb.close_project()

# Launch AEDT and import the configured EDB
aedt = pyaedt.Desktop("2022.1")
aedt.open_project(edb_path)

# Analyze DCIR
aedt.analyze_dcir()

# Retrieve and print loop resistance, current source, and via information
loop_resistance = aedt.get_dcir_element_data_loop_resistance()
current_source = aedt.get_dcir_element_data_current_source()
via = aedt.get_dcir_element_data_via()
print(f"Loop resistance: {loop_resistance}")
print(f"Current source: {current_source}")
print(f"Via: {via}")

# Retrieve and print voltage from the DCIR solution data
voltage = aedt.get_dcir_solution_data("VRM_Positive", "VRM_Negative")
print(f"Voltage: {voltage}")

# Close the AEDT project and release the desktop
aedt.close_project()
aedt.release_desktop()
```