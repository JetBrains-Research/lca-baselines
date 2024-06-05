 ```python
import logging
import dd4hep
from dd4hep import DD4hepSimulation
from dd4hep import get_code
from dd4hep import importConstants
from dd4hep import import_geometry
from dd4hep import enableDetailedHitsAndParticleInfo
from dd4hep import _set
from dd4hep import compileAClick
from dd4hep import get_unbound_function
from dd4hep import _setKernelProperty
from dd4hep import makeSet
from dd4hep import setOptions
from dd4hep import setOption
from dd4hep import run
from dd4hep import import_root
from dd4hep import import_detail
from dd4hep import _import_module
from dd4hep import import_rec
from dd4hep import import_tgeo
from dd4hep import _import_class
from dd4hep import import_units

def run_simulation():
    # Import additional modules
    _import_module("DDG4")
    _import_module("SimG4Core/Application")
    _import_module("SimG4Core/Notification")
    _import_module("SimG4Core/Interaction")
    _import_module("Geant4")

    # Set up the kernel
    _setKernelProperty("SimG4Core/Application/verbose", 1)
    _setKernelProperty("SimG4Core/Application/minimize", 1)
    _setKernelProperty("SimG4Core/Application/geant4-version", "10.03.p03")
    _setKernelProperty("SimG4Core/Application/g4beamline-version", "3.01")
    _setKernelProperty("SimG4Core/Application/physics-list", "QGSP_BERT_HP")
    _setKernelProperty("SimG4Core/Application/world-size", "10*mm")
    _setKernelProperty("SimG4Core/Action/counting", 1)

    # Load the geometry from a file
    geometry_path = "path/to/geometry/file.xml"
    det_factory = import_geometry(geometry_path)

    # Import constants
    importConstants()

    # Configure the Geant4 interface
    _setKernelProperty("SimG4Core/Notification/TrackingManager.Verbosity", 0)
    _setKernelProperty("SimG4Core/Notification/SteppingManager.Verbosity", 0)
    _setKernelProperty("SimG4Core/Notification/Kernel.Verbosity", 0)
    _setKernelProperty("SimG4Core/Interaction/G4EmStandardPhysics_option4", 1)
    _setKernelProperty("SimG4Core/Interaction/G4OpticalPhysics", 1)
    _setKernelProperty("SimG4Core/Interaction/G4Scintillation", 1)
    _setKernelProperty("SimG4Core/Interaction/G4ChargedGeantino", 1)
    _setKernelProperty("SimG4Core/Interaction/G4StoppingPhysics", 1)
    _setKernelProperty("SimG4Core/Interaction/G4DecayPhysics", 1)
    _setKernelProperty("SimG4Core/Interaction/G4IonPhysics", 1)
    _setKernelProperty("SimG4Core/Interaction/G4RadioactiveDecayPhysics", 1)
    _setKernelProperty("SimG4Core/Interaction/G4HadronElasticPhysicsHP", 1)
    _setKernelProperty("SimG4Core/Interaction/G4NeutronTrackingCut", 1)
    _setKernelProperty("SimG4Core/Interaction/G4ProcessTable.Verbosity", 0)

    # Set up the tracking field
    _import_class("DDG4", "Field")
    _import_class("DDG4", "ConstantMagneticField")
    _import_class("DDG4", "ConstantElectricField")
    _import_class("DDG4", "FieldManager")
    _import_class("DDG4", "SimpleSensitiveDetector")
    _import_class("DDG4", "DetectorElement")
    _import_class("DDG4", "World")
    _import_class("DDG4", "MaterialBudget")
    _import_class("DDG4", "ActionInitialization")
    _import_class("DDG4", "PhysicsList")
    _import_class("DDG4", "DetectorConstruction")
    _import_class("DDG4", "RunAction")
    _import_class("DDG4", "StackingAction")
    _import_class("DDG4", "EventAction")
    _import_class("DDG4", "PrimaryGeneratorAction")

    # Set up event actions
    _setKernelProperty("SimG4Core/Application/stack-size", 500)
    _setKernelProperty("SimG4Core/Application/max-stack-size", 1000)

    # Set up the particle gun
    _setKernelProperty("SimG4Core/Application/particle-gun", 1)
    _setKernelProperty("SimG4Core/Application/particle-gun-energy", 100.0)
    _setKernelProperty("SimG4Core/Application/particle-gun-theta", 0.0)
    _setKernelProperty("SimG4Core/Application/particle-gun-phi", 0.0)
    _setKernelProperty("SimG4Core/Application/particle-gun-polarization", 0.0)
    _setKernelProperty("SimG4Core/Application/particle-gun-pdg-code", 13)

    # Handle simulation particles
    _setKernelProperty("SimG4Core/Application/simulation-particles", [
        {
            "pdg-code": 11,
            "momentum": (0.0, 0.0, 10.0),
            "position": (0.0, 0.0, 0.0),
            "polarization": (0.0, 0.0, 0.0),
            "production-vertex": (0.0, 0.0, 0.0),
            "time": 0.0,
            "weight": 1.0
        }
    ])

    # Build the physics list
    _import_class("SimG4Core/PhysicsLists", "PhysicsList")
    physics_list = get_unbound_function("SimG4Core/PhysicsLists", "PhysicsList")()
    physics_list.SetVerbose(0)
    physics_list.RegisterPhysics(get_code("G4EmStandardPhysics_option4"))
    physics_list.RegisterPhysics(get_code("G4OpticalPhysics"))
    physics_list.RegisterPhysics(get_code("G4Scintillation"))
    physics_list.RegisterPhysics(get_code("G4ChargedGeantino"))
    physics_list.RegisterPhysics(get_code("G4StoppingPhysics"))
    physics_list.RegisterPhysics(get_code("G4DecayPhysics"))
    physics_list.RegisterPhysics(get_code("G4IonPhysics"))
    physics_list.RegisterPhysics(get_code("G4RadioactiveDecayPhysics"))
    physics_list.RegisterPhysics(get_code("G4HadronElasticPhysicsHP"))
    physics_list.RegisterPhysics(get_code("G4NeutronTrackingCut"))

    # Start the engine
    run(
        DetectorConstruction=det_factory,
        PhysicsList=physics_list,
        ActionInitialization=get_unbound_function("DDG4", "ActionInitialization")(),
        SimG4Core_Application=get_unbound_function("SimG4Core/Application", "Application")()
    )

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_simulation()
```