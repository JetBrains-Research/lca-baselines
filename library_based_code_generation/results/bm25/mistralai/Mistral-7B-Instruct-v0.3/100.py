 ```python
from dd4hep.core.kernel import Kernel
from dd4hep.visual.gui import GUI
from dd4hep.visual.view import View
from dd4hep.configuration.setup_util import setup_detector
from dd4hep.simulation.geant4 import Geant4, ParticleHandler
from dd4hep.simulation.tracker import Tracker
from dd4hep.simulation.physics_list import PhysicsList
from dd4hep.simulation.particle_gun import ParticleGun
from dd4hep.simulation.event_action import EventAction
from dd4hep.simulation.tracking_field import TrackingField
from dd4hep.util.findfiles import find_file

# Initialize kernel and load geometry from XML file
kernel = Kernel()
kernel.set_option("DD4hepExamplesINSTALL", ".")
setup_detector(kernel, "Detector.xml")

# Import constants from the kernel's detector description
constants = kernel.detector.Constants()

# Configure Geant4 instance
geant4 = Geant4()
geant4.enableDetailedHitsAndParticleInfo()
geant4.setDumpDetailedParticleInfo(True)

# Set up user interface
gui = GUI(kernel)
view = View(kernel)

# Configure tracking field
tracking_field = TrackingField(kernel)
tracking_field._setMagneticFieldOptions(constants.Bz)

# Set up event actions
event_action = EventAction()
event_action.set_option("PrintVerbose", True)

# Set up particle gun with gamma particle, energy of 5 keV, and multiplicity of 1
particle_gun = ParticleGun(geant4.particle_gun_name)
particle_gun.add_particle("gamma", 5e3)
particle_gun.set_multiplicity(1)

# Set up tracker
tracker = Tracker(kernel)
tracker.setupTracker()

# Set up physics list
physics_list = PhysicsList()
physics_list.add_particle_group("StandardModel")
physics_list.add_process("StandardModel")
physics_list.add_process("StandardModel:EM")
physics_list.add_process("StandardModel:HadronicInelastic")
physics_list.add_process("StandardModel:HadronicElastic")

# Configure user particle handler
user_particle_handler = ParticleHandler()
user_particle_handler.setupUserParticleHandler(kernel)

# Set up Geant4 instance with tracker and physics list
geant4_instance = geant4.with_metaclass(user_particle_handler)
geant4_instance.set_tracker(tracker)
geant4_instance.set_physics_list(physics_list)

# Raise Geant4 exception if any
raise_from(geant4_instance.compileAClick(), geant4_instance)

# Execute Geant4 instance
geant4_instance.execute()
```