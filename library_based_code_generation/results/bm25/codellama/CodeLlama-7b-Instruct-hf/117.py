  Here is some sample Python code that uses the blenderproc library to load a 3D scene from an .obj file and texture files, specified by command line arguments, and perform the desired operations:
```
import blenderproc
import argparse

# Parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--scene_file", type=str, help="Path to the .obj file containing the scene")
parser.add_argument("--texture_files", nargs="+", type=str, help="Paths to the texture files")
parser.add_argument("--output_dir", type=str, help="Path to the output directory")
args = parser.parse_args()

# Initialize the blenderproc library
blenderproc.init()

# Load the scene
scene = blenderproc.load_and_create(args.scene_file)

# Label the objects in the scene
mapping = {
    "wall": "Wall",
    "floor": "Floor",
    "ceiling": "Ceiling",
    "lamp": "Lamp",
    "ceiling_lamp": "Ceiling Lamp"
}
for obj in scene.objects:
    obj.label = mapping[obj.name]

# Separate walls, floors, and ceilings into distinct objects
walls = []
floors = []
ceilings = []
for obj in scene.objects:
    if obj.label == "Wall":
        walls.append(obj)
    elif obj.label == "Floor":
        floors.append(obj)
    elif obj.label == "Ceiling":
        ceilings.append(obj)

# Assign appropriate labels to the walls, floors, and ceilings
for obj in walls:
    obj.label = "Wall"
for obj in floors:
    obj.label = "Floor"
for obj in ceilings:
    obj.label = "Ceiling"

# Make lamp and ceiling objects emit light
for obj in scene.objects:
    if obj.label == "Lamp" or obj.label == "Ceiling Lamp":
        obj.emissive = True

# Create a bounding volume hierarchy (BVH) tree containing all objects in the scene
bvh = blenderproc.create_bvh_tree_multi_objects(scene.objects)

# Sample camera locations and rotations above the floor
camera_locations = []
camera_rotations = []
for i in range(10):
    camera_location = (0, 0, 10)
    camera_rotation = (0, 0, 0)
    camera_locations.append(camera_location)
    camera_rotations.append(camera_rotation)

# Check if the camera pose is valid and add it to the list if it is
valid_camera_poses = []
for camera_location, camera_rotation in zip(camera_locations, camera_rotations):
    if blenderproc.check_if_setup_utilities_are_at_the_top(scene, camera_location, camera_rotation):
        valid_camera_poses.append((camera_location, camera_rotation))

# Render the scene and write the data to a .hdf5 file in the specified output directory
blenderproc.render_scene(scene, valid_camera_poses, output_dir=args.output_dir)
```
Note that this is just sample code and may need to be modified to fit your specific use case. Additionally, you will need to have the blenderproc library installed and available in your Python environment in order to run this code.