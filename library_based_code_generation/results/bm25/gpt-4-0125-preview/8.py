```python
import argparse
import blenderproc as bproc
import numpy as np
import csv
import random

def load_label_mapping(csv_file):
    mapping = {}
    with open(csv_file, mode='r') as infile:
        reader = csv.reader(infile)
        mapping = {rows[0]: rows[1] for rows in reader}
    return mapping

parser = argparse.ArgumentParser(description='Initialize a Blender scene with blenderproc')
parser.add_argument('--scene', type=str, help='Path to the scene file')
parser.add_argument('--texture', type=str, help='Path to the texture directory')
parser.add_argument('--material', type=str, help='Path to the material file')
parser.add_argument('--output', type=str, help='Output directory')
parser.add_argument('--label_mapping', type=str, help='Path to the CSV file for label mapping')

args = parser.parse_args()

bproc.init()

# Load the scene
scene = bproc.loader.load_blend(args.scene)

# Load label mapping
label_mapping = load_label_mapping(args.label_mapping)

# Label objects
for obj in scene:
    if obj.get_name() in label_mapping:
        obj.set_cp("category_id", label_mapping[obj.get_name()])

# Load materials
materials = bproc.loader.load_and_create(args.material)

# Randomly assign materials to 40% of the objects
for obj in scene:
    if random.random() < 0.4:
        obj.set_material(random.choice(materials))

# Load textures for the materials
for material in materials:
    bproc.material.load_texture(material, args.texture)

# Extract floors and ceilings from walls and assign category IDs
bproc.object._assign_materials_to_floor_wall_ceiling(scene)

# Make lamp objects emit light
for obj in scene:
    if obj.get_type() == 'LIGHT':
        obj.make_emissive()

# Make ceiling objects emit a bit of light
for obj in scene:
    if "ceiling" in obj.get_name().lower():
        obj.make_emissive(strength=0.5)

# Create BVH tree and sample camera locations
bvh_tree = bproc.object.create_bvh_tree_multi_objects(bproc.object.get_all_mesh_objects())
camera_positions = bproc.camera.sample_locations_above_floor(bvh_tree, min_height=1.0, max_height=2.0)

# Ensure good scene coverage and no obstacles in front of the camera
valid_positions = []
for pos in camera_positions:
    if bproc.camera.scene_coverage_score(pos) > 0.5 and not bproc.camera.check_if_obstacle_in_view(pos):
        valid_positions.append(pos)

# Set camera to a valid position
bproc.camera.set_location(valid_positions[0])

# Enable normal, depth, and segmentation rendering
bproc.renderer.enable_normals_output()
bproc.renderer.enable_depth_output()
bproc.renderer.enable_segmentation_output()

# Render the scene
data = bproc.renderer.render()

# Write data to .hdf5 file
bproc.writer.write_hdf5(args.output, data)
```