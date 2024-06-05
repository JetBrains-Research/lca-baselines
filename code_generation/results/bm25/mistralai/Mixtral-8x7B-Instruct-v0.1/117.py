 import argparse
import os
import numpy as np

import blenderproc as bproc

def load\_scene(file\_path, texture\_dir, obj\_name, **kwargs):
bproc.init\_blenderproc()
check\_if\_setup\_utilities\_are\_at\_the\_top()

node = get\_the\_one\_node\_with\_type("FileLoader")
load\_and\_create(node, file\_path, **kwargs)

objects = scene.objects

# Label objects based on provided mapping
for obj in objects:
mapping = {
"sofa": "furniture",
"bed": "furniture",
"chair": "furniture",
"table": "furniture",
"tv\_stand": "furniture",
"lamp": "light",
"ceiling": "light",
"wall": "wall",
"floor": "floor",
}

material = obj.material\_slots[0].material
if material:
mat\_name = material.name
label = mapping.get(mat\_name.lower(), "other")
obj.blender\_object.data.name = f"{obj_name}_{label}"

# Separate walls, floors, and ceilings
wall\_objects = [obj for obj in objects if "wall" in obj.name.lower()]
floor\_objects = [obj for obj in objects if "floor" in obj.name.lower()]
ceiling\_objects = [obj for obj in objects if "ceiling" in obj.name.lower()]

# Assign appropriate labels
for obj in wall\_objects:
obj.blender\_object.data.name = obj.name.replace("floor", "wall")
obj.blender\_object.data.name = obj.name.replace("ceiling", "wall")
obj.label = "wall"

for obj in floor\_objects:
obj.label = "floor"

for obj in ceiling\_objects:
obj.label = "light"

# Make lamp and ceiling objects emit light
for obj in objects:
if "lamp" in obj.name.lower() or "ceiling" in obj.name.lower():
bproc.light\_suncg\_scene(obj, intensity=1200)

# Create a bounding volume hierarchy (BVH) tree containing all objects in the scene
create\_bvh\_tree\_multi\_objects(objects)

return objects

def sample\_camera\_poses(objects, num\_poses, floor\_height, **kwargs):
camera\_poses = []

for _ in range(num\_poses):
# Sample camera location above the floor
while True:
camera\_location = np.random.uniform(low=-5, high=5, size=3)
if not any(np.isclose(camera\_location, obj.location) for obj in objects if obj.label != "floor"):
break

# Sample camera rotation
camera\_rotation = np.random.uniform(low=0, high=2 * np.pi, size=1)

# Check if the camera pose is valid
if scene\_coverage\_score(camera\_location, camera\_rotation, floor\_height, objects) > 0.3:
camera\_poses.append((camera\_location, camera\_rotation))

return camera\_poses

def main(file\_path, texture\_dir, output\_dir, num\_poses, floor\_height):
os.makedirs(output\_dir, exist\_ok=True)

# Load the scene
objects = load\_scene(file\_path, texture\_dir, "scene", file\_format="obj", obj\_name\_prefix="", scene\_bbox\_scale=1.5)

# Sample camera poses
camera\_poses = sample\_camera\_poses(objects, num\_poses, floor\_height)

# Render the scene
for i, (camera\_location, camera\_rotation) in enumerate(camera\_poses):
bproc.set\_scene\_camera(location=camera\_location, rotation=(0, camera\_rotation, 0))
bproc.render(write\_images=False, output\_folder=output\_dir, image\_format="exr", image\_prefix=f"frame_{i:06d}")

# Write data to a .hdf5 file
data = {
"camera\_locations": np.array([pose[0] for pose in camera\_poses]),
"camera\_rotations": np.array([pose[1] for pose in camera\_poses]),
}

with h5py.File(os.path.join(output\_dir, "data.hdf5"), "w") as f:
for key, value in data.items():
f.create\_dataset(key, data=value)

if **kwargs.get("physics", False):
bproc.simulate\_physics\_and\_fix\_final\_poses(objects, iterations=100)
bproc.hide\_links\_and\_collision\_inertial\_objs(objects)

bproc.cleanup()

if **name** == "**main**":
parser = argparse.ArgumentParser()
parser.add\_argument("--file\_path", required=True, help="Path to the .obj file")
parser.add\_argument("--texture\_dir", required=True, help="Path to the directory containing texture files")
parser.add\_argument("--output\_dir", required=True, help="Path to the output directory")
parser.add\_argument("--num\_poses", type=int, default=10, help="Number of camera poses to sample (default: 10)")
parser.add\_argument("--floor\_height", type=float, default=0.0, help="Floor height (default: 0.0)")
parser.add\_argument("--physics", action="store\_true", help="Enable physics simulation")
args = parser.parse\_args()

main(args.file\_path, args.texture\_dir, args.output\_dir, args.num\_poses, args.floor\_height)