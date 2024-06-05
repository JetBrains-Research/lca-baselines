  from dd4hep import Detector, Material, Volume, Placement, Rotation, Compound, SensitiveDetector, Scintillator, LogicalVolume, PhysicalVolume, Geometry, MaterialPropertiesTable, ParticleGun, Particle, Energy, Multiplicity, Tracker, PhysicsList, Process, ParticleGroup, ParticleProcess, EventAction, UserInterface, TrackingField, Geant4

# Initialize kernel and load geometry from XML file
kernel = Detector()
kernel.load_geometry("DD4hepExamplesINSTALL/xml/example.xml")

# Import constants from detector description
constants = kernel.constants

# Set up Geant4 instance with tracker
geant4 = Geant4(kernel)
geant4.set_tracker(Tracker(geant4))

# Configure user interface
ui = UserInterface(geant4)

# Configure tracking field
tracking_field = TrackingField(geant4)

# Configure event actions
event_actions = EventAction(geant4)

# Set up particle gun
particle_gun = ParticleGun(geant4)
particle_gun.set_particle(Particle.gamma)
particle_gun.set_energy(Energy(5, "keV"))
particle_gun.set_multiplicity(Multiplicity(1))

# Set up tracker
tracker = Tracker(geant4)

# Set up physics list
physics_list = PhysicsList(geant4)
physics_list.add_particle_group(ParticleGroup.gamma)
physics_list.add_particle_process(ParticleProcess.gamma_conversion)
physics_list.add_particle_process(ParticleProcess.electromagnetic)

# Execute Geant4 instance
geant4.execute()