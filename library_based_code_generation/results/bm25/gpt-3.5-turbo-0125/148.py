import os
from pyaedt import Hfss
from pyaedt import Circuit
from pyaedt import Mechanical

hfss = Hfss()
hfss.set_graphical_mode()
hfss.open_project("path/to/project.aedt")
hfss.activate_ansys_design("HFSS", "2021 R2")
circuit = Circuit()
circuit.add_hfss_dynamic_link(hfss)
circuit.set_dynamic_link_options()
circuit.create_ports_and_excitations()
circuit.create_setup()
circuit.solve()
circuit.push_excitations_to_hfss()
mechanical = Mechanical()
mechanical.copy_bodies_from_hfss_project(hfss)
losses = hfss.get_losses()
mechanical.assign_convection(losses)
mechanical.plot_model()
mechanical.solve()
mechanical.plot_thermal_results()
hfss.release()