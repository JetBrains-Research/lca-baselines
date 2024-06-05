 ```python
import argparse
import logging
import dd4hep
from dd4hep import DD4hepDetector
from dd4hep import Physics
from dd4hep import geometry_builder
from dd4hep import QGSP_BERT_HP
from dd4hep import SimulationEngine
from dd4hep import VisualizationEngine
from dd4hep import ApplicationManager
from dd4hep import gigetracker
from dd4hep import conditions

def print_help():
    print("Usage: python simulation.py [options]")
    print("Options:")
    print("  -h, --help            Show this help message and exit")
    print("  -v, --visualize       Enable visualization")
    print("  -g, --geometry <file> Load a specific geometry from a file")

def main(args):
    # Set up logger
    logging.basicConfig(level=logging.INFO)

    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Run a simulation using dd4hep.")
    parser.add_argument("--visualize", action="store_true", help="Enable visualization")
    parser.add_argument("--geometry", type=str, help="Load a specific geometry from a file")
    args = parser.parse_args()

    # Set up Geant4 kernel and detector description
    det = DD4hepDetector("MyDetector")

    # Configure user interface
    gi5 = gigetracker.Gi5Tracker()
    gi5.Detector = det

    # Load geometry
    if args.geometry:
        det.merge(dd4hep.readDD4hepXML(args.geometry))

    # Set up magnetic field tracking
    det.field.setMagneticField("solenoid", 3.0, 0.0, 0.0)

    # Set up random generator
    rndm = conditions.LcgRandom()

    # Set up event actions
    event = SimulationEngine.SimulationAction()

    # Set up I/O
    io = SimulationEngine.SimulationIO()

    # Set up generator actions
    generator = SimulationEngine.SimulationGenerator()

    # Handle simulation particles
    generator.defineParticle("proton", 1, 1.00782503223)
    generator.defineParticle("pi+", 211, 0.13957018)
    generator.defineParticle("pi-", -211, 0.13957018)
    generator.defineParticle("electron", 11, 0.0005109989461)
    generator.defineParticle("gamma", 22, 0.0)

    # Setup detectors
    det_builder = geometry_builder.DetectorBuilder()
    det_builder.build(det)

    # Build the physics list
    phys = Physics.Simulation()
    phys.add(QGSP_BERT_HP)

    # Add special particle types and global range cut
    phys.addSpecial(det)
    phys.setGlobalRangeCut(10.0)

    # Create simulation application
    app = ApplicationManager()

    # Configure, initialize, run, and terminate the kernel
    app.init(dd4hep.runData, args.visualize)
    app.finalize()
    app.simulation().run(10)
    app.terminate()

if __name__ == "__main__":
    main(args)
```