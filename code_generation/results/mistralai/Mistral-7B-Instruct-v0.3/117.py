 Here is a Python script that uses the BlenderProc library to perform the tasks as described:

```python
import argparse
import os
import blenderbim
import bpy
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--scene', type=str, required=True)
parser.add_argument('--textures', type=str, nargs='+', required=True)
parser.add_argument('--output', type=str, required=True)
args = parser.parse_args()

blenderbim.register()
bpy.ops.wm.blenderbim_open(filepath=args.scene)

# Load textures
for texture in args.textures:
    bpy.ops.image.open(filepath=texture)

# Map objects and separate walls, floors, and ceilings
object_mapping = {
    'Wall': 'Mesh_001',
    'Floor': 'Mesh_002',
    'Ceiling': 'Mesh_003'
}

for label, name in object_mapping.items():
    obj = bpy.data.objects[name]
    if obj.type == 'MESH':
        obj.blenderbim_properties.label = label
        if label == 'Floor':
            bpy.ops.object.mode_set(mode='EDIT')
            bpy.ops.mesh.separate(type='SELECTED')
            bpy.ops.object.mode_set(mode='OBJECT')

# Set lamp and ceiling objects to emit light
for light in ['Lamp_001', 'Ceiling_001']:
    obj = bpy.data.objects[light]
    if obj.type == 'MESH':
        obj.data.emissive_strength = 10

# Create BVH tree
bpy.ops.object.mode_set(mode='EDIT')
bpy.ops.mesh.select_all(action='DESELECT')
for obj in bpy.data.objects:
    if obj.blenderbim_properties.label:
        obj.select_set(True)
bpy.ops.object.bvh_tree_create()
bpy.ops.object.mode_set(mode='OBJECT')

# Sample camera locations and rotations
camera_locations = []
camera_rotations = []
for _ in range(10):
    bpy.context.view_layer.objects.active = bpy.data.objects['Camera']
    bpy.ops.object.location_clear()
    bpy.ops.object.rotation_clear()
    bpy.context.view_layer.objects.active = None

    bbox = bpy.data.objects['BVH_Tree'].bound_box
    center = bbox[0] + (bbox[1] - bbox[0]) / 2
    height = max(bbox[3][1] - bbox[0][1], bbox[3][2] - bbox[0][2])
    radius = max(bbox[1][0] - bbox[0][0], bbox[1][1] - bbox[0][1], bbox[1][2] - bbox[0][2])
    theta = np.random.uniform(0, 2 * np.pi)
    phi = np.random.uniform(0, np.pi)
    x = radius * np.cos(phi) * np.cos(theta)
    y = height + radius * np.sin(phi)
    z = radius * np.cos(phi) * np.sin(theta)

    bpy.context.view_layer.objects.active = bpy.data.objects['Camera']
    bpy.context.view_layer.objects.active.location = (x, y, z)
    bpy.context.view_layer.objects.active.rotation_euler = (0, theta, 0)

    # Check if there are no obstacles in front of the camera and the scene coverage score is not too low
    # (This requires additional code to calculate the camera frustum and check for intersections)

    if not any(obstacle for obstacle in bpy.context.view_layer.objects.intersect_all(context.view_layer.active_layer, obstacle=bpy.context.view_layer.objects)):
        camera_locations.append(bpy.context.view_layer.objects.active.location)
        camera_rotations.append(bpy.context.view_layer.objects.active.rotation_euler)

# Add cameras to the scene
for i, (location, rotation) in enumerate(zip(camera_locations, camera_rotations)):
    bpy.ops.object.camera_add(location=location, rotation=rotation)

# Enable normal, depth, and segmentation rendering
bpy.context.scene.render.layers[0].use_pass_normal = True
bpy.context.scene.render.layers[0].use_pass_depth = True
bpy.context.scene.render.layers[0].use_pass_zbuffer = True
bpy.context.scene.render.layers[0].use_pass_segmentation = True

# Render the scene and write the data to a .hdf5 file
bpy.ops.render.render(write_still=True, filepath=f"{args.output}/render.exr")
bpy.ops.export_scene.hdf5(filepath=f"{args.output}/scene.hdf5")
```

This script assumes that the BlenderProc library is properly installed and the scene file, textures, and output directory are provided as command line arguments. The script also assumes that there is a camera object named 'Camera' in the scene. Additionally, it requires additional code to calculate the camera frustum and check for intersections to ensure there are no obstacles in front of the camera and the scene coverage score is not too low.