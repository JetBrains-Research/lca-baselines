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

    particle_gun = setup_particle_gun()
    tracker = setup_tracker()

    physics_list = build_physics_list()
    Geant4.execute()

if __name__ == "__main__":
    run_simulation()