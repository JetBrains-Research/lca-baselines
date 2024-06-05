  Here is the Python code that sets up a dd4hep simulation using the provided APIs:
```
import logging
from dd4hep.simulation import DD4hepSimulation
from dd4hep.get_code import get_code
from dd4hep.importConstants import importConstants
from dd4hep.import_geometry import import_geometry
from dd4hep.enableDetailedHitsAndParticleInfo import enableDetailedHitsAndParticleInfo
from dd4hep.set import _set
from dd4hep.compileAClick import compileAClick
from dd4hep.get_unbound_function import get_unbound_function
from dd4hep.setKernelProperty import _setKernelProperty
from dd4hep.makeSet import makeSet
from dd4hep.setOptions import setOptions
from dd4hep.setOption import setOption
from dd4hep.run import run
from dd4hep.import_root import import_root
from dd4hep.import_detail import import_detail
from dd4hep._import_module import _import_module
from dd4hep.import_rec import import_rec
from dd4hep.import_tgeo import import_tgeo
from dd4hep._import_class import _import_class
from dd4hep.import_units import import_units

# Set up logging
logging.basicConfig(level=logging.INFO)

# Define function to run simulation
def run_simulation():
    # Import necessary modules
    import_root()
    import_detail()
    import_rec()
    import_tgeo()
    import_units()

    # Set up kernel
    _setKernelProperty("Geant4", "Geant4")

    # Load geometry from file
    import_geometry("path/to/geometry.gdml")

    # Import constants
    importConstants()

    # Configure Geant4 interface
    _set("Geant4", "Geant4")

    # Set up tracking field
    _set("TrackingField", "TrackingField")

    # Set up event actions
    _set("EventActions", "EventActions")

    # Set up particle gun
    _set("ParticleGun", "ParticleGun")

    # Handle simulation particles
    _set("SimulationParticles", "SimulationParticles")

    # Build physics list
    _set("PhysicsList", "PhysicsList")

    # Start engine
    run()

# Run simulation if script is run as main program
if __name__ == "__main__":
    run_simulation()
```
This code imports the necessary modules, sets up logging, defines a function to run the simulation, and runs the simulation if the script is run as the main program. The function sets up the kernel, loads the geometry from a file, imports constants, and configures the Geant4 interface. It also sets up the tracking field, event actions, and particle gun, and handles simulation particles, builds the physics list, and starts the engine.