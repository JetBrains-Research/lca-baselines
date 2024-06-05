 ```python
import pyaedt

EDA_Tool = pyaedt.Hfss3dLayout()
EDA_Tool.new_project()
EDA_Tool.set_working_directory(EDA_Tool.create_temp_directory())
EDA_Tool.open_example_project(EDA_Tool.dcir_example_project())

vrm\_pins = EDA\_Tool.get_pins\_from\_net('VRM\_PWR')
sink\_pins = EDA\_Tool.get_pins\_from_net('Sink\_PWR')

vrm\_pos\_pins = [pin for pin in vrm\_pins if 'POS' in pin.Name]
vrm\_neg\_pins = [pin for pin in vrm\_pins if 'NEG' in pin.Name]
sink\_pos\_pins = [pin for pin in sink\_pins if 'POS' in pin.Name]
sink\_neg\_pins = [pin for pin in sink\_pins if 'NEG' in pin.Name]

EDA\_Tool.test_134_create_port_between_pin_and_layer(vrm\_pos\_pins[0], 'Top', 'Port1')
EDA\_Tool.test_134_create_port_between_pin_and_layer(vrm\_neg_pins[0], 'Top', 'Port2')
EDA\_Tool.test_134_create_port_between_pin_and_layer(sink\_pos\_pins[0], 'Top', 'Port3')
EDA\_Tool.test_134_create_port_between_pin_and_layer(sink\_neg\_pins[0], 'Top', 'Port4')

EDA\_Tool.create_voltage_source_on_pin('Port1', 'Port2')
EDA\_Tool.create_current_source_on_pin('Port3', 'Port4')

EDA\_Tool.save_project()
EDA\_Tool.close_project()

aedt = pyaedt.Aedt()
aedt.new_project()
aedt.open_project(EDA_Tool.working_directory + '\\' + EDA_Tool.last_project)

analysis = aedt.analyze_design('DCIR')
analysis.solve()

dcir_data = analysis.get_data('DCIR')
loop_resistance = dcir_data.get_dcir_element_data_loop_resistance()
current_source = dcir_data.get_dcir_element_data_current_source()
via_data = dcir_data.get_dcir_element_data_via()
voltage = dcir_data.get_dcir_solution_data()

print('Loop Resistance:', loop_resistance)
print('Current Source:', current_source)
print('Via Data:', via_data)
print('Voltage:', voltage)

aedt.close_project()
pyaedt.clean_cache_and_files()
```