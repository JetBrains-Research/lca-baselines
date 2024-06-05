 Here is a simplified Python code using the pyaedt library for the given instruction. Note that you need to have the pyaedt library installed and the necessary credentials for HFSS 3D Layout.

```python
from pyaedt import *

# Configure EDB for DCIR analysis
temp_dir = os.path.join(os.getcwd(), 'temp')
os.makedirs(temp_dir, exist_ok=True)
edb_file = os.path.join(temp_dir, 'example_board.edb')
download_example_board(edb_file)

# Initialize AEDT session
session = Session()
session.open(edb_file)

# Create pin groups
vrm_pos_group = session.create_pin_group('VRM_POS')
vrm_neg_group = session.create_pin_group('VRM_NEG')
sink_pos_group = session.create_pin_group('SINK_POS')
sink_neg_group = session.create_pin_group('SINK_NEG')

# Create voltage source and current source
v_source = session.create_voltage_source(vrm_pos_group, vrm_neg_group, 1.2)
i_source = session.create_current_source(sink_pos_group, sink_neg_group, 100e-3)

# Add SIwave DCIR analysis
siwave_dcir = session.create_siwave_dcir(vrm_pos_group, vrm_neg_group, sink_pos_group, sink_neg_group)

# Save and close EDB
session.save()
session.close()

# Launch AEDT, import the configured EDB and analyze DCIR
aedt = AEDT()
aedt.open(edb_file)
aedt.run_analysis(siwave_dcir)

# Retrieve and print DCIR element data
loop_resistance = aedt.get_dcir_element_data('LoopResistance')[0]
current_source = aedt.get_dcir_element_data('CurrentSource')[0]
via_info = aedt.get_dcir_element_data('ViaInfo')[0]
print(f'Loop Resistance: {loop_resistance} Ohms')
print(f'Current Source: {current_source} A')
print(f'Via Info: {via_info}')

# Retrieve and print DCIR solution data
voltage = aedt.get_dcir_solution_data('Voltage')[0]
print(f'Voltage: {voltage} V')

# Close the AEDT project and release the desktop
aedt.close()
aedt.release_desktop()
```

This code assumes that you have an example board with VRM and sink components in the provided EDB file. Adjust the code according to your specific project requirements.