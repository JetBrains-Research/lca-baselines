 Here is a basic structure of the Python code using dd4hep and some other necessary libraries. Please note that this is a simplified version and might need adjustments based on your specific requirements.

```python
import argparse
from dd4hep.configuration import Configuration
from dd4hep.detector.hepml import HepML
from dd4hep.visualization.base import Visualization
from dd4hep.visualization.core import Canvas3D
from dd4hep.visualization.services import ViewerService
from dd4hep.simulation.kernel import Simulation
from dd4hep.simulation.services import (
    RandomEngine,
    EventAction,
    SteppingAction,
    TrackingAction,
    GeneratorAction,
    IOParser,
    IOService,
)
from dd4hep.simulation.utils import (
    ParticleTypes,
    Detector,
    MagneticField,
    GlobalRangeCut,
)

def print_help():
    print("Usage: python simulator.py [options]")
    print("Options:")
    print("-v, --visualize: Enable visualization")
    print("-g, --geometry: Geometry file path")
    print("-m, --magnetic-field: Magnetic field file path")
    print("-o, --output: Output file path")

def main(args):
    # Set up logger
    Configuration.instance().set_log_level("INFO")

    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--visualize", action="store_true")
    parser.add_argument("-g", "--geometry", type=str, required=True)
    parser.add_argument("-m", "--magnetic-field", type=str)
    parser.add_argument("-o", "--output", type=str, required=True)
    args = parser.parse_args(args)

    # Set up Geant4 kernel and detector description
    detector = HepML(filename=args.geometry)

    # Configure magnetic field tracking
    if args.magnetic_field:
        magnetic_field = MagneticField(filename=args.magnetic_field)
        detector.set_magnetic_field(magnetic_field)

    # Set up random generator
    random_engine = RandomEngine()

    # Configure event actions
    event_action = EventAction()

    # Configure stepping action
    stepping_action = SteppingAction()

    # Configure tracking action
    tracking_action = TrackingAction()

    # Configure generator actions
    generator_action = GeneratorAction()
    generator_action.add_particle(ParticleTypes.eplus, 1.0)
    generator_action.set_global_range_cut(GlobalRangeCut(0.01, 10.0))

    # Add special particle types
    # ...

    # Setup detectors
    # ...

    # Build physics list
    physics_list = detector.physics_list()

    # Configure user interface
    ui = detector.ui()

    # Configure I/O
    io_service = IOService()
    io_service.set_input_output(IOParser(args.output))

    # Configure visualization
    if args.visualize:
        visualization = Visualization()
        canvas = Canvas3D()
        viewer_service = ViewerService(visualization, canvas)
        ui.add_service(viewer_service)

    # Initialize simulation
    simulation = Simulation(detector, physics_list, ui, random_engine, event_action, stepping_action, tracking_action, generator_action, io_service)

    # Run simulation
    simulation.execute()

    # Terminate simulation
    simulation.terminate()

if __name__ == "__main__":
    print_help()
    main(sys.argv[1:])
```

This code sets up a basic simulation using dd4hep, parses command line arguments, and configures the necessary services for the simulation. However, it does not include the setup for special particle types, additional detectors, or any specific geometry or magnetic field files. You will need to add those according to your requirements. Also, the visualization part is included but not fully functional, as it requires additional setup for the specific visualization you want to use.