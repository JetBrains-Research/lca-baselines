  import pyaedt

# Configure EDB for DCIR analysis
edb = pyaedt.Edb(edbpath)
edb.create_temp_directory()
edb.download_example_board("board_name")

# Create pin groups on VRM positive and negative pins
vrm_pos_pins = edb.get_pin_list("VRM_positive")
vrm_neg_pins = edb.get_pin_list("VRM_negative")
edb.create_pin_group("VRM_pins", vrm_pos_pins + vrm_neg_pins)

# Create voltage source between VRM pin groups
edb.create_voltage_source("VRM_pins", "VRM_voltage", "DC")

# Create pin groups on sink component positive and negative pins
sink_pos_pins = edb.get_pin_list("sink_component_positive")
sink_neg_pins = edb.get_pin_list("sink_component_negative")
edb.create_pin_group("sink_pins", sink_pos_pins + sink_neg_pins)

# Create current source between sink pin groups
edb.create_current_source("sink_pins", "sink_current", "DC")

# Add SIwave DCIR analysis
edb.add_analysis("siwave_dcir")

# Save and close EDB
edb.save_project()
edb.close_project()

# Launch AEDT and import EDB
aedt = pyaedt.Desktop("2022.1")
aedt.open_project(edbpath)

# Analyze DCIR
aedt.analyze_setup("siwave_dcir")

# Retrieve loop resistance, current source, and via information from DCIR element data
loop_resistance = aedt.get_siwave_dcir_element_data("loop_resistance")
current_source = aedt.get_siwave_dcir_element_data("current_source")
via_info = aedt.get_siwave_dcir_element_data("via_info")

# Retrieve voltage from DCIR solution data
voltage = aedt.get_siwave_dcir_solution_data("voltage")

# Print loop resistance, current source, via information, and voltage
print("Loop resistance:", loop_resistance)
print("Current source:", current_source)
print("Via information:", via_info)
print("Voltage:", voltage)

# Close AEDT project and release desktop
aedt.close_project()
aedt.release_desktop()