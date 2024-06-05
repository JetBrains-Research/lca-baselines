```python
import dd4hep
import dd4hep.simulation as sim
import logging

def setup_logging():
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger()
    return logger

def run_simulation():
    import argparse
    import dd4hep.g4 as g4
    from dd4hep import Simulation
    
    parser = argparse.ArgumentParser(description='Run DD4hep simulation.')
    parser.add_argument('--geometry', type=str, help='Path to the geometry file', required=True)
    parser.add_argument('--help', action='store_true', help='Print help message')
    args = parser.parse_args()
    
    if args.help:
        parser.print_help()
        exit(0)
    
    logger = setup_logging()
    logger.info("Starting simulation setup...")
    
    dd4hep.setLoaderPath()
    
    # Load geometry
    geometry = dd4hep.Geometry(args.geometry)
    logger.info("Loaded geometry from %s", args.geometry)
    
    # Load constants
    dd4hep.loadConstants()
    
    # Setup Geant4
    geant4 = g4.Geant4(geometry)
    
    # Print detectors
    geant4.printDetectors()
    
    # Configure UI
    geant4.setupUI()
    
    # Tracking field
    geant4.setupTrackingField()
    
    # Event actions
    geant4.setupEventActions()
    
    # Particle gun
    geant4.setupParticleGun()
    
    # Tracker
    geant4.setupTracker()
    
    # Physics list
    geant4.setupPhysicsList()
    
    # Execute Geant4
    geant4.execute()
    logger.info("Simulation completed.")

if __name__ == "__main__":
    run_simulation()
```