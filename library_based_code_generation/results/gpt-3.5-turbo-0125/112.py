import logging
import dd4hep
from dd4hep import Simulation

def run_simulation():
    import dd4hep.kernel as kernel
    kernel.loadGeometry("path/to/geometry/file")
    
    import dd4hep.constants as constants
    
    dd4hep.setupGeant4()
    
    tracking_field = dd4hep.TrackingField()
    event_actions = dd4hep.EventActions()
    particle_gun = dd4hep.ParticleGun()
    
    dd4hep.handleSimulationParticles()
    dd4hep.buildPhysicsList()
    dd4hep.startEngine()

if __name__ == "__main__":
    run_simulation()