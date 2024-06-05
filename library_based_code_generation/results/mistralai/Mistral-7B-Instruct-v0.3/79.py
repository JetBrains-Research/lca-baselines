 Here is the Python code that follows your instructions:

```python
import os
import dd
from dd.configuration import Configuration
from dd.run.manager import Run
from dd4hep.detector.helper import Detector
from dd4hep.vis.opengl import OpenGLVisualizer
from geant4.kernel import Kernel
from geant4.kernel.geometry_factory import GeometryFactory
from geant4.physics_list import QGSP_BERT
from geant4.simulation.process_creator import ProcessCreator

def setup_simulation():
    # Initialize DDG4 kernel
    kernel = Kernel()

    # Load geometry from XML file
    geometry_file = os.path.join(os.environ['DD4hepExamplesINSTALL'], 'OpticalSurfaces/compact', 'compact.xml')
    detector = Detector(geometry_file)

    # Import constants from the kernel's detector description
    detector.process_description(kernel)

    # Configure Geant4 instance
    geant4 = Run(kernel)

    # Set up tracking field
    geant4.setup_tracking()

    # Set up event actions
    geant4.setup_event_action()

    # Set up detector construction
    geant4.setup_detector_construction(detector)

    # Set up particle gun
    particle_gun = geant4.get_primary_generator()
    particle_gun.set_particle("gamma")
    particle_gun.set_particle_energy(5 * 1000)  # 5 keV
    particle_gun.set_particle_multiplicity(1)

    # Set up tracker
    tracker_name = 'MaterialTester'
    tracker = geant4.get_detector(tracker_name)

    # Set up physics list
    physics_list = QGSP_BERT()
    geant4.set_physics_list(physics_list)

    # Set up UI with macro if provided
    ui = geant4.get_ui_manager()
    if __name__ == "__main__" and len(sys.argv) > 1:
        ui.apply_command('macro "' + sys.argv[1] + '"')

    # Execute Geant4 instance
    geant4.execute()

if __name__ == "__main__":
    setup_simulation()
```

This code assumes that you have the necessary Geant4 and dd4hep packages installed. Please make sure to replace the path to the geometry file if it is different in your setup. Also, the macro path in the UI setup should be adjusted according to your specific needs.