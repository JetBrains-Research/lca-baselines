 Here is the Python code that sets up a dd4hep simulation as per your instructions:

```python
import logging
from dd4hep.core.dd4hep import DD4hepSimulation
from dd4hep.configuration.builder import Builder
from dd4hep.configuration.storage import Storage
from dd4hep.detector.hepml import HepML
from dd4hep.utils.services import get_code, _set
from dd4hep.utils.units import MeV, cm
from dd4hep.simulation.kernel import Kernel
from dd4hep.simulation.geometry import Geometry
from dd4hep.simulation.constants import importConstants
from dd4hep.simulation.geant4 import enableDetailedHitsAndParticleInfo, _setKernelProperty, compileAClick
from dd4hep.simulation.tracking import TrackingField
from dd4hep.simulation.actions import EventAction
from dd4hep.simulation.particle_gun import ParticleGun
from dd4hep.simulation.physics_list import PhysicsList
from dd4hep.simulation.engine import Engine
from dd4hep.rec.root import import_root
from dd4hep.rec.detail import import_detail
from dd4hep.rec.tgeo import import_tgeo
from dd4hep.rec.units import import_units
from dd4hep.rec.rec import import_rec

def run_simulation():
    # Import additional modules
    _import_module('dd4hep.configuration.standard_lhc')
    _import_module('dd4hep.configuration.standard_materials')
    _import_module('dd4hep.configuration.standard_detectors')

    # Set up the kernel
    kernel = Kernel()
    _setKernelProperty(kernel, 'DetectorDescription', 'DD4hep')

    # Load the geometry from a file
    geometry = Geometry(kernel)
    builder = Builder(storage=Storage())
    builder.append(HepML('path_to_your_geometry_file.xml'))
    builder.build(geometry)

    # Import constants
    importConstants(kernel)

    # Configure the Geant4 interface
    enableDetailedHitsAndParticleInfo(kernel)
    compileAClick(kernel)

    # Set up the tracking field
    tracking_field = TrackingField(kernel)

    # Set up event actions
    event_action = EventAction()

    # Set up the particle gun
    particle_gun = ParticleGun(kernel)

    # Handle simulation particles
    particles = [('e-', 10*MeV)]

    # Build the physics list
    physics_list = PhysicsList(kernel)

    # Start the engine
    engine = Engine(kernel, tracking_field, event_action, particle_gun, physics_list)

    # Set up logging
    logging.basicConfig(level=logging.INFO)

    # Run the simulation
    engine.execute(particles)

if __name__ == "__main__":
    run_simulation()
```

Please replace `'path_to_your_geometry_file.xml'` with the actual path to your geometry file. This code sets up a basic dd4hep simulation with a particle gun, event actions, and a simple physics list. You may need to adjust the configuration according to your specific needs.