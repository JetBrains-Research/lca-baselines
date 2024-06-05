 import dd4hep
import logging
import sys
import argparse
from dd4hep import DD4hepSimulation
from dd4hep.opt import create_command_line_processor
from dd4hep.cli.tools import show\_help
from dd4hep.det import Detector
from dd4hep.geometry import GeometryVolume as gv
from dd4hep.geometry import Layer, Tube, Box, PlacedVolume
from dd4hep.segmentation import Section, SectionLayer, Rectangular, Regular
from dd4hep.material import vacuum, Be, Al, Si, Air, Cu
from dd4hep.func import FieldValue, FieldRegion, RotationY, RotationZ, Translation
from dd4hep.hepUnits import Length, Volume, Time, Energy, SolidAngle
from dd4hep.detdesc import DetElement, VolumeManager, SensitiveDetector, Simulation, Placement
from dd4hep.tpc import TPC
from dd4hep.simulation import SimulationFactory
from dd4hep.tuning import compileAClick
from Configurables import Geant4, SetPrintLevel, SetOptions, makeSet, SetOption

def run\_simulation(args):
if args.help:
show\_help(sys.argv)
sys.exit(0)

import dd4hep
import dd4hep.det
import dd4hep.geometry
import dd4hep.segmentation
import dd4hep.material
import dd4hep.func
import dd4hep.detdesc
import dd4hep.tpc
import dd4hep.simulation
import dd4hep.tuning

# Import command line arguments
parser = argparse.ArgumentParser(description='Simulation with dd4hep.')
parser.add\_argument('--geom', type=str, required=True, help='Geometry file')
args = parser.parse\_args(args)

# Import geometry
geom = dd4hep.det.Detector.getInstance()
geom.loadGeometry(args.geom)

# Import constants
dd4hep.importConstants()

# Print detectors
printDetectors(geom)

# Set up Geant4
g4 = Geant4()
g4.setUI(True)
g4.setDumpDetailedParticleInfo(True)
g4.setOptions(g4.options().setProperty("SimG4Core/Kernel/verbose", 1))
g4.setOptions(g4.options().setProperty("SimG4Core/Kernel/debug", 0))
g4.setOptions(g4.options().setProperty("SimG4Core/Kernel/debugVerbose", 0))
g4.setOptions(g4.options().setProperty("SimG4Core/Logging/LogLevel", 1))
g4.setOptions(g4.options().setProperty("SimG4Core/MessageLogger/TrackingManager", 1))
g4.setOptions(g4.options().setProperty("SimG4Core/MessageLogger/RunManager", 1))
g4.setOptions(g4.options().setProperty("SimG4Core/MessageLogger/Visualization", 1))
g4.setOptions(g4.options().setProperty("SimG4Core/MessageLogger/Transportation", 1))
g4.setOptions(g4.options().setProperty("SimG4Core/MessageLogger/StackingAction", 1))
g4.setOptions(g4.options().setProperty("SimG4Core/MessageLogger/EventAction", 1))
g4.setOptions(g4.options().setProperty("SimG4Core/MessageLogger/SteppingAction", 1))
g4.setOptions(g4.options().setProperty("SimG4Core/MessageLogger/Navigation", 1))
g4.setOptions(g4.options().setProperty("SimG4Core/MessageLogger/SensitiveDetector", 1))
g4.setOptions(g4.options().setProperty("SimG4Core/MessageLogger/Physics", 1))
g4.setOptions(g4.options().setProperty("SimG4Core/MessageLogger/DetectorConstruction", 1))
g4.setOptions(g4.options().setProperty("SimG4Core/MessageLogger/RunAction", 1))
g4.setOptions(g4.options().setProperty("SimG4Core/MessageLogger/ApplicationState", 1))
g4.setOptions(g4.options().setProperty("SimG4Core/MessageLogger/Initialization", 1))
g4.setOptions(g4.options().setProperty("SimG4Core/MessageLogger/Command", 1))
g4.setOptions(g4.options().setProperty("SimG4Core/MessageLogger/StartOfEventAction", 1))
g4.setOptions(g4.options().setProperty("SimG4Core/MessageLogger/EndOfEventAction", 1))
g4.setOptions(g4.options().setProperty("SimG4Core/MessageLogger/BeginOfRunAction", 1))
g4.setOptions(g4.options().setProperty("SimG4Core/MessageLogger/EndOfRunAction", 1))
g4.setOptions(g4.options().setProperty("SimG4Core/MessageLogger/BeginOfSimulationAction", 1))
g4.setOptions(g4.options().setProperty("SimG4Core/MessageLogger/EndOfSimulationAction", 1))
g4.setOptions(g4.options().setProperty("SimG4Core/MessageLogger/SteppingVerbose", 1))
g4.setOptions(g4.options().setProperty("SimG4Core/MessageLogger/TrackingVerbose", 1))
g4.setOptions(g4.options().setProperty("SimG4Core/MessageLogger/PhysicsVerbose", 1))
g4.setOptions(g4.options().setProperty("SimG4Core/MessageLogger/DetectorConstructionVerbose", 1))
g4.setOptions(g4.options().setProperty("SimG4Core/MessageLogger/RunActionVerbose", 1))
g4.setOptions(g4.options().setProperty("SimG4Core/MessageLogger/ApplicationStateVerbose", 1))
g4.setOptions(g4.options().setProperty("SimG4Core/MessageLogger/InitializationVerbose", 1))
g4.setOptions(g4.options().setProperty("SimG4Core/MessageLogger/CommandVerbose", 1))
g4.setOptions(g4.options().setProperty("SimG4Core/MessageLogger/StartOfEventActionVerbose", 1))
g4.setOptions(g4.options().setProperty("SimG4Core/MessageLogger/EndOfEventActionVerbose", 1))
g4.setOptions(g4.options().setProperty("SimG4Core/MessageLogger/BeginOfRunActionVerbose", 1))
g4.setOptions(g4.options().setProperty("SimG4Core/MessageLogger/EndOfRunActionVerbose", 1))
g4.setOptions(g4.options().setProperty("SimG4Core/MessageLogger/BeginOfSimulationActionVerbose", 1))
g4.setOptions(g4.options().setProperty("SimG4Core/MessageLogger/EndOfSimulationActionVerbose", 1))
g4.setOptions(g4.options().setProperty("SimG4Core/MessageLogger/SteppingVerbose", 1))
g4.setOptions(g4.options().setProperty("SimG4Core/MessageLogger/TrackingVerbose", 1))
g4.setOptions(g4.options().setProperty("SimG4Core/MessageLogger/PhysicsVerbose", 1))
g4.setOptions(g4.options().setProperty("SimG4Core/MessageLogger/DetectorConstructionVerbose", 1))
g4.setOptions(g4.options().setProperty("SimG4Core/MessageLogger/RunActionVerbose", 1))
g4.setOptions(g4.options().setProperty("SimG4Core/MessageLogger/ApplicationStateVerbose", 1))
g4.setOptions(g4.options().setProperty("SimG4Core/MessageLogger/InitializationVerbose", 1))
g4.setOptions(g4.options().setProperty("SimG4Core/MessageLogger/CommandVerbose", 1))
g4