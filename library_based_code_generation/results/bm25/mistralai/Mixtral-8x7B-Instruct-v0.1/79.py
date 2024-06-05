 import os
import sys
import ddg4
from ddg4 import Geant4, TrackingField, EventActions, DetectorConstruction, Gun, UI
from ddg4.core import _setKernelProperty, setOption, makeSet, addDetectorConstruction
from ddg4.simulation import compileAClick
from ddg4.particles import ParticleHandler
from dd4hep import DD4hepKernel
from dd4hep.optics import Optics
from dd4hep.detdesc import Detector

def runSimulation():
 dd4hepKernel = DD4hepKernel("DD4hepExamplesINSTALL")
 dd4hepKernel.initialize(DD4hepKernel.jobType.Simulation)
 dd4hepKernel.loadGeometry(os.path.join(os.getenv("DD4hepExamplesINSTALL"), "OpticalSurfaces/compact/opticalsurfaces.xml"))
 detector = dd4hepKernel.detector()

 # Import constants from the kernel's detector description
 dd4hepKernel.simulation().importConstantsFromDetector(detector)

 # Configure Geant4 instance
 field = TrackingField()
 geant4 = Geant4(
 Field=field,
 EventActions=EventActions,
 DetectorConstruction=DetectorConstruction(detector),
 Gun=Gun(
 Particle=ParticleHandler.gamma,
 Energy=5.0 * ddg4.keV,
 Multiplicity=1
 ),
 UI=UI(
 CommandLine=CommandLine(
 AClickFile=compileAClick("gun.macro") if len(sys.argv) > 1 else None
 )
 )
 )

 # Set up a tracker named 'MaterialTester' and a physics list named 'QGSP_BERT'
 geant4.setOption("SimG4Core/Tracking/Verbosity", 0)
 geant4.setOption("SimG4Core/Kernel/Verbosity", 0)
 geant4.setOption("SimG4Core/Physics/Verbosity", 0)
 geant4.setOption("SimG4Core/Application/Verbosity", 0)
 geant4.setOption("SimG4Core/Process/Verbosity", 0)
 geant4.setOption("SimG4Em/Verbosity", 0)
 geant4.setOption("SimG4Em/Physics/Verbosity", 0)
 geant4.setOption("SimG4EmProcesses/Verbosity", 0)
 geant4.setOption("SimG4EmProcesses/EmStandardPhysics/Verbosity", 0)
 geant4.setOption("SimG4EmProcesses/EmExtraPhysics/Verbosity", 0)
 geant4.setOption("SimG4EmProcesses/EmLivermorePhysics/Verbosity", 0)
 geant4.setOption("SimG4EmProcesses/EmPenelopePhysics/Verbosity", 0)
 geant4.setOption("SimG4EmProcesses/EmQGSP_BERT/Verbosity", 0)
 geant4.setOption("SimG4EmProcesses/EmG4DecayPhysics/Verbosity", 0)
 geant4.setOption("SimG4EmProcesses/EmG4IonPhysics/Verbosity", 0)
 geant4.setOption("SimG4EmProcesses/EmG4StoppingPhysics/Verbosity", 0)
 geant4.setOption("SimG4EmProcesses/EmG4HadronPhysicsQGSP_BERT/Verbosity", 0)
 geant4.setOption("SimG4EmProcesses/EmG4HadronPhysicsQGSP_BIC_AllCharged/Verbosity", 0)
 geant4.setOption("SimG4EmProcesses/EmG4HadronPhysicsQGSP_BIC/Verbosity", 0)
 geant4.setOption("SimG4EmProcesses/EmG4HadronPhysicsQGSP_BERT_HP/Verbosity", 0)
 geant4.setOption("SimG4EmProcesses/EmG4HadronPhysicsQGSP_BIC_HP/Verbosity", 0)
 geant4.setOption("SimG4EmProcesses/EmG4HadronElasticPhysics/Verbosity", 0)
 geant4.setOption("SimG4EmProcesses/EmG4CapturePhysics/Verbosity", 0)
 geant4.setOption("SimG4EmProcesses/EmG4PionPhysics/Verbosity", 0)
 geant4.setOption("SimG4EmProcesses/EmG4KaonPhysics/Verbosity", 0)
 geant4.setOption("SimG4EmProcesses/EmG4ProtonPhysics/Verbosity", 0)
 geant4.setOption("SimG4EmProcesses/EmG4AntiProtonPhysics/Verbosity", 0)
 geant4.setOption("SimG4EmProcesses/EmG4DeuteronPhysics/Verbosity", 0)
 geant4.setOption("SimG4EmProcesses/EmG4TritonPhysics/Verbosity", 0)
 geant4.setOption("SimG4EmProcesses/EmG4AlphaPhysics/Verbosity", 0)
 geant4.setOption("SimG4EmProcessics/EmG4GenericIonPhysics/Verbosity", 0)
 geant4.setOption("SimG4EmProcesses/EmG4CapturePhysicsHP/Verbosity", 0)
 geant4.setOption("SimG4EmProcesses/EmG4PhotoElectricPhysics/Verbosity", 0)
 geant4.setOption("SimG4EmProcesses/EmG4ComptonPhysics/Verbosity", 0)
 geant4.setOption("SimG4EmProcesses/EmG4RayleighPhysics/Verbosity", 0)
 geant4.setOption("SimG4EmProcesses/EmG4PairProductionPhysics/Verbosity", 0)
 geant4.setOption("SimG4EmProcesses/EmG4BremsstrahlungPhysics/Verbosity", 0)
 geant4.setOption("SimG4EmProcesses/EmG4IonisationPhysics/Verbosity", 0)
 geant4.setOption("SimG4EmProcesses/EmG4MultiSamplingProcess/Verbosity", 0)
 geant4.setOption("SimG4EmProcesses/EmG4ScintillationPhysics/Verbosity", 0)
 geant4.setOption("SimG4EmProcesses/EmG4ScintillationPhysicsHP/Verbosity", 0)
 geant4.setOption("SimG4EmProcesses/EmG4CerenkovPhysics/Verbosity", 0)
 geant4.setOption("SimG4EmProcesses/EmG4MieHGPhysics/Verbosity", 0)
 geant4.setOption("SimG4EmProcesses/EmG4GasEmPhysics/Verbosity", 0)
 geant4.setOption("SimG4EmProcesses/EmG4LivermorePolarisedComptonPhysics/Verbosity", 0)
 geant4.setOption("SimG4EmProcesses/EmG4G4DecayPhysics/Verbosity", 0)
 geant4.setOption("SimG4EmProcesses/EmG4G4IonPhysics/Verbosity", 0)
 geant4.setOption("SimG4EmProcesses/EmG4G4StoppingPhysics/Verbosity", 0)
 geant4.setOption("SimG4EmProcesses/EmG4G4HadronPhysicsQGSP_BERT/Verbosity", 0)
 geant4.setOption("SimG4EmProcesses/EmG4G4HadronPhysicsQGSP_BIC_AllCharged/Verbosity", 0)
 geant4.setOption("SimG4EmProcesses/EmG4G4HadronPhysicsQGSP_BIC/Verbosity", 0)
 geant4.setOption("SimG4EmProcesses/EmG4G4HadronPhysicsQGSP_BERT_HP/Verbosity", 0)
 ge