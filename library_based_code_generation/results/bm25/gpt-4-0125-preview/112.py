```python
import logging
from DD4hep import DD4hepSimulation, importConstants, import_geometry, enableDetailedHitsAndParticleInfo
from DD4hepSimulation import _set, compileAClick, get_unbound_function, _setKernelProperty, makeSet, setOptions, setOption, run
from DD4hep import import_root, import_detail, _import_module, import_rec, import_tgeo, _import_class, import_units

def run_simulation():
    import logging
    from DD4hep import Simulation, Geometry
    from DD4hep.Geometry import VolumeManager
    from DD4hep.Simulation import Geant4Kernel, Geant4Config
    from SystemOfUnits import mm, cm, GeV
    import Geant4Interfaces
    
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger('dd4hep')

    kernel = Geant4Kernel.Instance()
    geometry_file = "path/to/your/geometry/file"
    import_geometry(kernel, geometry_file)
    importConstants(kernel)
    
    Geant4Config.setup()
    enableDetailedHitsAndParticleInfo(kernel)
    
    # Configure the Geant4 interface
    g4interface = Geant4Interfaces.Geant4(kernel)
    g4interface.configure()
    
    # Setup tracking field
    _setKernelProperty(kernel, "Field", "MagneticField")
    
    # Setup event actions
    eventAction = _import_class("DDG4.EventAction")
    kernel.registerGlobalAction(eventAction())
    
    # Setup the particle gun
    particleGun = _import_class("DDG4.ParticleGun")
    kernel.registerGlobalAction(particleGun())
    particleGun().particle = "e-"
    particleGun().energy = 10 * GeV
    particleGun().position = (0, 0, -10) * cm
    
    # Handle simulation particles
    particleHandler = _import_class("DDG4.ParticleHandler")
    kernel.registerGlobalAction(particleHandler())
    
    # Build the physics list
    physicsList = _import_class("G4PhysListFactory").getPhysicsList("FTFP_BERT")
    kernel.setPhysicsList(physicsList)
    
    # Start the engine
    run(kernel, numberOfEvents=100)

if __name__ == "__main__":
    run_simulation()
```