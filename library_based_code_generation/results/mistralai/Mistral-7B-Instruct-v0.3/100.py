 ```python
from dd4hep.configuration import Configuration
from dd4hep.detector.hepml import HepMLDetector
from dd4hep.visualization.hepvis import HepVisApp
from dd4hep.simulation.geant4 import Geant4Kernel, Geant4Tracker
from dd4hep.simulation.geant4.processes import Geant4StandardPhysicsList
from dd4hep.simulation.particles import ParticleGun
from dd4hep.utils.xml_input import XmlInput

# Initialize a kernel and load geometry from XML file
kernel = Geant4Kernel()
detector = HepMLDetector(name='MyDetector', configuration=Configuration(XmlInput('{DD4hepExamplesINSTALL}/config/detector/simple_detector.xml')))
kernel.setDetector(detector)

# Import constants from the kernel's detector description
constants = kernel.detector.constants()

# Set up Geant4 instance with a tracker
geant4 = kernel.geant4()
geant4.setTracker(Geant4Tracker(detector))

# Configure user interface, tracking field, and event actions
geant4.userInterface().visualization().enable()
geant4.trackingField().enable()
geant4.eventAction().enable()

# Set up particle gun with a gamma particle, energy of 5 keV, and multiplicity of 1
gun = ParticleGun(particleType='Gamma', position=[0, 0, 0], energy=5*1e-3, multiplicity=1)
geant4.eventAction().setParticleGun(gun)

# Set up tracker and physics list, adding various particle groups and processes
geant4.tracker().addPhysicsList(Geant4StandardPhysicsList())

# Execute the Geant4 instance
geant4.execute()

# Set up visualization
app = HepVisApp(detector)
app.run()
```