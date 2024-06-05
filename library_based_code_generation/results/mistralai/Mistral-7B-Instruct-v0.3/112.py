 Here is a basic example of a Python script that sets up a dd4hep simulation using the given instructions. Please note that this is a simplified example and you may need to adjust it according to your specific needs.

```python
import sys
import logging
from dd4hep.configuration import Builder, Configuration
from dd4hep.detector.hep3dbool import Hep3DBool
from dd4hep.detector.hep3dvector import Hep3DVector
from dd4hep.simulation.kernel import Simulation
from dd4hep.simulation.detector import Detector
from dd4hep.simulation.particle_gun import ParticleGun
from dd4hep.simulation.action import ActionInitialisation, ActionPrimaryVerbose
from dd4hep.simulation.physics.em Standard_Model import QED

def run_simulation():
    # Create a new builder
    builder = Builder()

    # Set up logging
    logging.basicConfig(level=logging.INFO)

    # Define the geometry
    builder.include("dd4hep/detector/ecal.py")
    builder.include("dd4hep/detector/hcal.py")

    # Import constants
    builder.include("dd4hep/constants.py")

    # Configure the Geant4 interface
    builder.use_detector(Detector)
    builder.use_kernel(Simulation)

    # Set up the tracking field
    builder.add_parameter("DD4hep.Tracking.Field", "use_magnetic_field", Hep3DBool(True))

    # Set up event actions
    builder.add_action(ActionInitialisation())
    builder.add_action(ActionPrimaryVerbose())

    # Set up the particle gun
    gun = builder.add_node("ParticleGun", "particleGun")
    gun.add_property("ParticleType", "e-")
    gun.add_property("Position", Hep3DVector(0, 0, 0))
    gun.add_property("Momentum", Hep3DVector(1, 0, 0))

    # Handle simulation particles
    particles = builder.add_node("Particles", "particles")
    particles.add_property("ParticleGun", "particleGun")
    particles.add_property("NumberOfParticles", 1000)

    # Build the physics list
    builder.add_physics(QED)

    # Start the engine
    configuration = Configuration(builder.build())
    configuration.execute()

if __name__ == "__main__":
    run_simulation()
```

This script sets up a simple simulation with an ECal and HCal detector, a particle gun that shoots electrons, and the QED physics list. The simulation runs for 1000 particles. You may need to adjust the detector geometry, particle properties, and physics list according to your specific needs.