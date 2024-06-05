```python
from dd4hep import DD4hepSimulation

simulation = DD4hepSimulation()
simulation.initialize_kernel()
simulation.load_geometry_from_xml_file('path/to/XML/file.xml')
simulation.import_constants_from_detector_description()
simulation.setup_geant4_instance_with_tracker()
simulation.configure_user_interface()
simulation.configure_tracking_field()
simulation.configure_event_actions()
simulation.setup_particle_gun(particle='gamma', energy=5, multiplicity=1)
simulation.setup_tracker()
simulation.setup_physics_list(particle_groups=['gamma', 'e-', 'e+', 'proton'], processes=['compton', 'phot', 'conv', 'msc'])
simulation.execute_geant4_instance()
```