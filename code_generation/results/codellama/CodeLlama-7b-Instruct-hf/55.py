  import dd4hep

def display_help():
    print("Usage: python simulation.py [options]")
    print("Options:")
    print("  -h, --help                  Display this help message and exit")
    print("  -g, --geometry <file>       Load geometry from file")
    print("  -m, --magnetic-field <file>  Load magnetic field from file")
    print("  -i, --interactive            Enable interactive mode")
    print("  -v, --visualize              Enable visualization")

def main():
    # Set up logger
    logger = dd4hep.logger.Logger()

    # Parse command line arguments
    args = dd4hep.arguments.ArgumentParser()
    args.add_argument("-g", "--geometry", help="Load geometry from file")
    args.add_argument("-m", "--magnetic-field", help="Load magnetic field from file")
    args.add_argument("-i", "--interactive", action="store_true", help="Enable interactive mode")
    args.add_argument("-v", "--visualize", action="store_true", help="Enable visualization")
    args.parse_args()

    # Set up Geant4 kernel and detector description
    kernel = dd4hep.kernel.Kernel()
    detector = dd4hep.detector.Detector()

    # Configure user interface
    if args.interactive:
        kernel.configure_user_interface(dd4hep.user_interface.UserInterface.kInteractive)
    else:
        kernel.configure_user_interface(dd4hep.user_interface.UserInterface.kBatch)

    # Load geometry
    if args.geometry:
        detector.load_geometry(args.geometry)

    # Load magnetic field
    if args.magnetic_field:
        detector.load_magnetic_field(args.magnetic_field)

    # Configure random generator
    kernel.configure_random_generator(dd4hep.random.RandomGenerator.kDefault)

    # Set up event actions
    kernel.add_event_action(dd4hep.event_action.EventAction.kSimulation)
    kernel.add_event_action(dd4hep.event_action.EventAction.kTracking)

    # Set up I/O
    kernel.configure_io(dd4hep.io.Io.kDefault)

    # Set up various generator actions
    kernel.add_generator_action(dd4hep.generator_action.GeneratorAction.kSimulation)
    kernel.add_generator_action(dd4hep.generator_action.GeneratorAction.kTracking)

    # Set up simulation particles
    kernel.add_simulation_particle(dd4hep.simulation_particle.SimulationParticle.kGamma)
    kernel.add_simulation_particle(dd4hep.simulation_particle.SimulationParticle.kElectron)
    kernel.add_simulation_particle(dd4hep.simulation_particle.SimulationParticle.kPositron)

    # Set up detectors
    kernel.add_detector(detector)

    # Build physics list
    kernel.build_physics_list(dd4hep.physics_list.PhysicsList.kStandard)

    # Add special particle types
    kernel.add_special_particle_type(dd4hep.special_particle_type.SpecialParticleType.kGamma)
    kernel.add_special_particle_type(dd4hep.special_particle_type.SpecialParticleType.kElectron)
    kernel.add_special_particle_type(dd4hep.special_particle_type.SpecialParticleType.kPositron)

    # Set global range cut
    kernel.set_global_range_cut(1000)

    # Configure visualization
    if args.visualize:
        kernel.configure_visualization(dd4hep.visualization.Visualization.kDefault)

    # Configure, initialize, run, and terminate the kernel
    kernel.configure()
    kernel.initialize()
    kernel.run()
    kernel.terminate()

if __name__ == "__main__":
    main()