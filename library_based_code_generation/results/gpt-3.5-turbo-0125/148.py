import os
from pyaedt import Hfss
from pyaedt import Circuit
from pyaedt import Mechanical

hfss = Hfss()
hfss.version = "2021.1"

circuit = Circuit()
circuit.add_hfss_dynamic_link(hfss)
circuit.set_dynamic_link_options()
circuit.create_ports_and_excitations()
circuit.create_setup()
circuit.solve()
circuit.push_excitations_to_hfss()

mechanical = Mechanical()
mechanical.copy_bodies_from_hfss_project()
losses = hfss.get_losses()
mechanical.assign_convection(losses)

mechanical.plot_model()
mechanical.solve()
mechanical.plot_thermal_results()

hfss.release()