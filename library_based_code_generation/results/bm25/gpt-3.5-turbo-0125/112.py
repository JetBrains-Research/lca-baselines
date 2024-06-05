import dd4hep
import logging

def run_simulation():
    import dd4hep.kernel
    import dd4hep.sim
    import dd4hep.field
    import dd4hep.event
    import dd4hep.particle

    kernel = dd4hep.kernel.Kernel()
    kernel.loadGeometry("geometry_file.xml")
    constants = dd4hep.importConstants()
    dd4hep.enableDetailedHitsAndParticleInfo()
    kernel._set("tracking_field", dd4hep.field.TrackingField())
    kernel._set("event_actions", dd4hep.event.EventActions())
    kernel._set("particle_gun", dd4hep.particle.ParticleGun())

    simulation_particles = dd4hep.get_code()
    physics_list = dd4hep.makeSet()
    engine = dd4hep.run()

if __name__ == "__main__":
    run_simulation()