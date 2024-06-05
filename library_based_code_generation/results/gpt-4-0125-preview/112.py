```python
import dd4hep
import dd4hep.simulation as sim
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run_simulation():
    from dd4hep import Simulation, Kernel
    from dd4hep import Geometry
    from dd4hep.simulation import geant4 as g4
    from dd4hep.core.constants import mm, GeV, tesla

    # Setup the kernel
    kernel = Kernel("DD4hepSim")
    
    # Load the geometry
    geometry_file = "path/to/your/geometry/file"
    geometry = Geometry(geometry_file)
    kernel.loadGeometry(geometry)

    # Configure the Geant4 interface
    geant4 = g4.Geant4Kernel(kernel)
    geant4.loadGeometry(geometry)

    # Setup tracking field
    field = g4.ConstantField()
    field.magneticField = (0, 0, 4*tesla)  # Example: 4 Tesla magnetic field in Z direction
    geant4.add(field)

    # Setup event actions
    event_action = g4.EventAction()
    geant4.add(event_action)

    # Setup the particle gun
    particle_gun = g4.ParticleGun()
    particle_gun.particle = "e-"
    particle_gun.energy = 10*GeV
    particle_gun.position = (0, 0, -1*mm)
    particle_gun.direction = (0, 0, 1)
    geant4.addPrimaryGenerator(particle_gun)

    # Handle simulation particles
    # Example: Registering a particle handler for electrons
    particle_handler = g4.ParticleHandler()
    particle_handler.addParticle("e-")
    geant4.add(particle_handler)

    # Build the physics list
    physics_list = g4.FTFP_BERT()
    geant4.setPhysicsList(physics_list)

    # Start the engine
    geant4.initialize()
    geant4.run(10)  # Run 10 events for example
    geant4.terminate()

if __name__ == "__main__":
    run_simulation()
```