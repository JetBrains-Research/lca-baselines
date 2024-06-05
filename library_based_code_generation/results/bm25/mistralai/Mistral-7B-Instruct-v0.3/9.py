 Here is the Python code based on your instructions:

```python
import sys
import argparse
from dd4hep.core.api import CompileAClick, enableDetailedHitsAndParticleInfo, DD4hepSimulation, _set, get_unbound_function, setPrintLevel, setOptions, makeSet, setOption
from dd4hep.configuration.api import Geant4, __setMagneticFieldOptions
from dd4hep.utils.api import importConstants, printDetectors, _setKernelProperty
from dd4hep.io.hepml import import_geometry
from dd4hep.detector.api import printDetector
from dd4hep.simulation.api import setDumpDetailedParticleInfo
from dd4hep.cli.api import show_help, is_package, get_code

def run_simulation(args):
    parser = argparse.ArgumentParser(description='Run a simulation with dd4hep')
    parser.add_argument('geometry', help='Geometry file')
    parser.add_argument('--help', action='store_true', help='Show help')

    args = parser.parse_args(args)

    if args.help:
        show_help(parser)
        sys.exit(0)

    if not is_package('dd4hep'):
        print('dd4hep not found, please install it.')
        sys.exit(1)

    setPrintLevel(2)
    simulation = DD4hepSimulation()

    # Import constants
    constants = importConstants('dd4hep.units')
    _setKernelProperty('constants', constants)

    # Set up Geant4
    Geant4()

    # Print detectors
    printDetectors()

    # Configure UI, tracking field, and event actions
    simulation.ui().setOption('GUI', False)
    __setMagneticFieldOptions(simulation, 'off')
    setDumpDetailedParticleInfo(True)

    # Set up particle gun and tracker
    gun = simulation.gun()
    gun.setParticle('e-')
    gun.setPosition(0, 0, 0)
    gun.setEnergy(100 * constants.MeV)

    tracker = simulation.tracker()

    # Build physics list and execute Geant4
    physics_list = simulation.physicsList()
    physics_list.add('QGSP_BERT', 'QGSP_BERT_HP', 'QGSP_BERT_EM', 'QGSP_BERT_EM_EX', 'QGSP_BERT_EM_INEL', 'QGSP_BERT_EM_INEL_HP', 'QGSP_BERT_EM_INEL_HP_EX', 'QGSP_BERT_EM_INEL_HP_EM', 'QGSP_BERT_EM_INEL_HP_EM_EX', 'QGSP_BERT_EM_INEL_HP_EM_INEL', 'QGSP_BERT_EM_INEL_HP_EM_INEL_HP', 'QGSP_BERT_EM_INEL_HP_EM_INEL_HP_EX', 'QGSP_BERT_EM_INEL_HP_EM_INEL_HP_EM', 'QGSP_BERT_EM_INEL_HP_EM_INEL_HP_EM_EX', 'QGSP_BERT_EM_INEL_HP_EM_INEL_HP_EM_INEL', 'QGSP_BERT_EM_INEL_HP_EM_INEL_HP_EM_INEL_HP', 'QGSP_BERT_EM_INEL_HP_EM_INEL_HP_EM_INEL_HP_EX', 'QGSP_BERT_EM_INEL_HP_EM_INEL_HP_EM_INEL_HP_EM', 'QGSP_BERT_EM_INEL_HP_EM_INEL_HP_EM_INEL_HP_EM_EX', 'QGSP_BERT_EM_INEL_HP_EM_INEL_HP_EM_INEL_HP_EM_INEL', 'QGSP_BERT_EM_INEL_HP_EM_INEL_HP_EM_INEL_HP_EM_INEL_HP', 'QGSP_BERT_EM_INEL_HP_EM_INEL_HP_EM_INEL_HP_EM_INEL_HP_EX', 'QGSP_BERT_EM_INEL_HP_EM_INEL_HP_EM_INEL_HP_EM_INEL_HP_EM', 'QGSP_BERT_EM_INEL_HP_EM_INEL_HP_EM_INEL_HP_EM_INEL_HP_EM_EX', 'QGSP_BERT_EM_INEL_HP_EM_INEL_HP_EM_INEL_HP_EM_INEL_HP_EM_INEL', 'QGSP_BERT_EM_INEL_HP_EM_INEL_HP_EM_INEL_HP_EM_INEL_HP_EM_INEL_HP', 'QGSP_BERT_EM_INEL_HP_EM_INEL_HP_EM_INEL_HP_EM_INEL_HP_EM_INEL_HP_EX', 'QGSP_BERT_EM_INEL_HP_EM_INEL_HP_EM_INEL_HP_EM_INEL_HP_EM_INEL_HP_EM', 'QGSP_BERT_EM_INEL_HP_EM_INEL_HP_EM_INEL_HP_EM_INEL_HP_EM_INEL_HP_EM_EX', 'QGSP_BERT_EM_INEL_HP_EM_INEL_HP_EM_INEL_HP_EM_INEL_HP_EM_INEL_HP_EM_INEL', 'QGSP_BERT_EM_INEL_HP_EM_INEL_HP_EM_INEL_HP_EM_INEL_HP_EM_INEL_HP_EM_INEL_HP', 'QGSP_BERT_EM_INEL_HP_EM_INEL_HP_EM_INEL_HP_EM_INEL_HP_EM_INEL_HP_EM_INEL_HP_EX', 'QGSP_BERT_EM_INEL_HP_EM_INEL_HP_EM_INEL_HP_EM_INEL_HP_EM_INEL_HP_EM_INEL_HP_EM', 'QGSP_BERT_EM_INEL_HP_EM_INEL_HP_EM_INEL_HP_EM_INEL_HP_EM_INEL_HP_EM_INEL_HP_EM_EX', 'QGSP_BERT_EM_INEL_HP_EM_INEL_HP_EM_INEL_HP_EM_INEL_HP_EM_INEL_HP_EM_INEL_HP_EM_INEL', 'QGSP_BERT_EM_INEL_HP_EM_INEL_HP_EM_INEL_HP_EM_INEL_HP_EM_INEL_HP_EM_INEL_HP_EM_INEL_HP', 'QGSP_BERT_EM_INEL_HP_EM_INEL_HP_EM_INEL_HP_EM_INEL_HP_EM_INEL_HP_EM_INEL_HP_EM_INEL_HP_EX', 'QGSP_BERT_EM_INEL_HP_EM_INEL_HP_EM_INEL_HP_EM_INEL_HP_EM_INEL_HP_EM_INEL_HP_EM_INEL_HP_EM', 'QGSP_BERT_EM_INEL_HP_EM_INEL_HP_EM_INEL_HP_EM_INEL_HP_EM_INEL_HP_EM_INEL_HP_EM_INEL_HP_EM_EX', 'QGSP_BERT_EM_INEL_HP_EM_INEL_HP_EM_INEL_HP_EM_INEL_HP_EM_INEL_HP_EM_INEL_HP_EM_INEL_HP_EM_INEL', 'QGSP_BERT_EM_INEL_HP_EM_INEL_HP_EM_INEL_HP_EM_INEL_HP_EM_INEL_HP_EM_INEL_HP_EM_INEL_HP_EM