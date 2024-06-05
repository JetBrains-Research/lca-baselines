```python
import dd4hep
import dd4hep.simulation as sim
import argparse
import sys

def display_help():
    print("Usage: python simulation.py [options]")
    print("Options:")
    print("  -h, --help            show this help message and exit")
    print("  -g, --geometry=FILE   specify the geometry file to use")
    print("  -n, --num-events=N    specify the number of events to simulate")
    print("  --enable-visualization  enable visualization")

def main():
    parser = argparse.ArgumentParser(description='Run a DD4hep simulation.')
    parser.add_argument('-g', '--geometry', required=True, help='Geometry file')
    parser.add_argument('-n', '--num-events', type=int, default=10, help='Number of events to simulate')
    parser.add_argument('--enable-visualization', action='store_true', help='Enable visualization')
    args = parser.parse_args()

    # Setup logger
    dd4hep.setPrintLevel(dd4hep.INFO)

    # Setup the DD4hep simulation kernel
    kernel = sim.G4Kernel()
    kernel.loadGeometry('file:' + args.geometry)

    # Setup magnetic field and tracking
    field = sim.setupMagneticField(kernel)
    sim.setupTracking(kernel, field)

    # Setup random generator
    sim.setupRandomGenerator(kernel)

    # Setup event actions, I/O, and generator actions
    sim.setupEventActions(kernel)
    sim.setupIO(kernel)
    sim.setupGeneratorAction(kernel)

    # Setup simulation particles and detectors
    sim.setupParticles(kernel)
    sim.setupDetectors(kernel)

    # Build the physics list and add special particle types and global range cut
    physicsList = sim.setupPhysicsList(kernel, 'FTFP_BERT')
    sim.addSpecialParticleTypes(kernel)
    sim.setGlobalRangeCut(kernel, 0.7)

    # Configure and initialize the kernel
    kernel.initialize()
    kernel.applyUICommand('/run/initialize')

    # Enable visualization if requested
    if args.enable_visualization:
        sim.setupVisualization(kernel)
        kernel.applyUICommand('/vis/open OGL 600x600-0+0')
        kernel.applyUICommand('/vis/drawVolume')
        kernel.applyUICommand('/vis/viewer/set/viewpointThetaPhi 90 90')
        kernel.applyUICommand('/vis/viewer/zoom 1.4')

    # Run the simulation
    kernel.runSimulation(args.num_events)

    # Terminate the kernel
    kernel.terminate()

if __name__ == '__main__':
    main()
```