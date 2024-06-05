 Here is a basic structure of the Python code using dd4hep library. Please note that this is a simplified version and you may need to adjust it according to your specific needs.

```python
import argparse
from dd4hep.core.api import (
    get_code,
    DD4hepSimulation,
    compileAClick,
    enableDetailedHitsAndParticleInfo,
    _kernel_terminate,
    addParametersToRunHeader,
    GeneratorAction,
    CommandLine,
    run,
    commandsPostRun,
    commandsPreRun,
    kernel,
    mapActions,
    commandsInitialize,
    commandsTerminate,
    MagneticField,
    get_unbound_function,
    _configureDD4HEP,
    setupTrackingField,
    setupUserPhysics
)

def print_help():
    print("Usage: python simulator.py [options]")
    print("Options:")
    print("-h, --help            Show this help message and exit")
    print("-v, --visualize       Enable visualization")

def main(args):
    _configureDD4HEP()

    # Set up logger
    logger = get_code().getLogger("dd4hep")

    # Parse command line arguments
    parser = argparse.ArgumentParser(description="DD4hep Simulation")
    parser.add_argument("-v", "--visualize", action="store_true", help="Enable visualization")
    args = parser.parse_args(args)

    # Set up Geant4 kernel and detector description
    sim = DD4hepSimulation()

    # Configure the user interface
    sim.ui.setDefaultVisualization(args.visualize)

    # Load a specific geometry
    # ... (load your geometry here)

    # Set up magnetic field tracking
    mf = MagneticField()
    mf.setWorldMagneticField(get_unbound_function("worldMagneticField"))
    setupTrackingField(sim, mf)

    # Set up random generator
    # ... (set up your random generator here)

    # Set up event actions
    # ... (set up your event actions here)

    # Set up I/O
    # ... (set up your I/O here)

    # Set up various generator actions
    # ... (set up your generator actions here)

    # Setup detectors
    # ... (set up your detectors here)

    # Build the physics list
    # ... (build your physics list here)

    # Add special particle types and a global range cut
    # ... (add your special particle types and range cut here)

    # Configure user physics
    setupUserPhysics(sim)

    # Configure, initialize, run, and terminate the kernel
    sim.configure()
    sim.initialize()
    sim.run()
    _kernel_terminate()

if __name__ == "__main__":
    main(CommandLine().parse_args())
```

This code provides a basic structure for a DD4hep simulation. You will need to fill in the missing parts according to your specific requirements. For example, you will need to load your geometry, set up your random generator, event actions, I/O, generator actions, detectors, and physics list. Additionally, you may need to add more specific configurations for your simulation.