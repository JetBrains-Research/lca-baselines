# Configure EDB for DCIR analysis
temp_dir = os.path.join(os.getcwd(), "temp_dir")
if not os.path.exists(temp_dir):
    os.makedirs(temp_dir)
aedtapp = HFSS3DLayout(edbpath=temp_dir)
aedtapp.download_example("example_board")

# Create pin groups on VRM positive and negative pins
aedtapp.create_pin_group("VRM_positive_pins", ["VRM_pos1", "VRM_pos2"])
aedtapp.create_pin_group("VRM_negative_pins", ["VRM_neg1", "VRM_neg2"])

# Create voltage source between VRM pin groups
aedtapp.create_voltage_source("VRM_positive_pins", "VRM_negative_pins")

# Create pin groups on sink component positive and negative pins
aedtapp.create_pin_group("sink_positive_pins", ["sink_pos1", "sink_pos2"])
aedtapp.create_pin_group("sink_negative_pins", ["sink_neg1", "sink_neg2"])

# Place current source between sink component pin groups
aedtapp.create_current_source("sink_positive_pins", "sink_negative_pins")

# Add SIwave DCIR analysis
aedtapp.add_siwave_dcir_analysis()

# Save and close EDB
aedtapp.save_edb()
aedtapp.close_edb()

# Launch AEDT, import configured EDB and analyze DCIR
aedtapp = HFSS3DLayout(edbpath=temp_dir)
aedtapp.analyze_dcir()

# Retrieve and print loop resistance, current source, via information and voltage
loop_resistance = aedtapp.get_dcir_element_data("loop_resistance")
current_source = aedtapp.get_dcir_element_data("current_source")
via_info = aedtapp.get_dcir_element_data("via_info")
voltage = aedtapp.get_dcir_solution_data("voltage")
print("Loop Resistance:", loop_resistance)
print("Current Source:", current_source)
print("Via Information:", via_info)
print("Voltage:", voltage)

# Close AEDT project and release desktop
aedtapp.close_project()
aedtapp.release_desktop()