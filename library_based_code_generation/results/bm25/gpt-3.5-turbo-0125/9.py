import logging
import sys
import argparse
import dd4hep

def run_simulation():
    import additional_modules
    import command_line_arguments
    import geometry_file

    if args.help:
        print("Help message")
        sys.exit()

    import constants
    import Geant4
    import detectors

    configure_UI()
    configure_tracking_field()
    configure_event_actions()

    set_up_particle_gun()
    set_up_tracker()

    build_physics_list()
    execute_Geant4()

if __name__ == "__main__":
    run_simulation()