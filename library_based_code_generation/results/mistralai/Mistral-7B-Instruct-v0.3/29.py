 ```python
import numpy as np
import pyvista
from pyvista import Plottable
from ament_ros_perception_msgs.srv import GetModel
import rospy

def get_model(name):
    rospy.wait_for_service('get_model')
    get_model = rospy.ServiceProxy('get_model', GetModel)
    return get_model(name).models

def load_planet(name):
    model = get_model(name)
    polydata = pyvista.PolyData(model)
    texture = pyvista.Textures.from_file(f"{name}_texture.png")
    polydata.texture = texture
    return polydata

def solar_system_visualization():
    solar_system = pyvista.Plotter()

    # Load and position planets
    sun = load_planet("sun")
    sun.translate([0, 0, 0])
    solar_system.add_subplot(sun)

    earth = load_planet("earth")
    earth.translate([-1, 0, 0])
    earth_subplot = solar_system.add_subplot(earth)
    earth_subplot.view_parallel(azimuth=135, elevation=30)

    mars = load_planet("mars")
    mars.translate([1, 0, 0])
    mars_subplot = solar_system.add_subplot(mars)
    mars_subplot.view_parallel(azimuth=45, elevation=30)

    # Load Venus with and without atmosphere
    venus = load_planet("venus")
    venus_no_atmosphere = load_planet("venus_no_atmosphere")
    venus.translate([0.75, 0, 0])
    venus_no_atmosphere.translate([0.75, 0, 0.1])
    venus_subplot = solar_system.add_subplot(venus)
    venus_no_atmosphere_subplot = solar_system.add_subplot(venus_no_atmosphere)
    venus_subplot.view_parallel(azimuth=225, elevation=30)
    venus_no_atmosphere_subplot.view_parallel(azimuth=225, elevation=30)

    # Set light source
    light = pyvista.LightSource(2, 2, -1)
    solar_system.light = light

    solar_system.show()

if __name__ == "__main__":
    solar_system_visualization()
```

Please note that this code assumes that you have the models of the planets and their textures in the correct format (.obj or .vtk) and in the same directory as the script, with the naming convention of `planet_name.obj` or `planet_name.vtk` for the models and `planet_name_texture.png` for the textures.

Also, this code uses ROS services to load the models. You need to have a ROS node that provides these services. The code provided here is just a simplified example and might need adjustments to fit your specific use case.