 ```python
import argparse
import os
import bpy
import blenderproc
from blenderproc.publicTypes.materialTypes import Material
from blenderproc.publicTypes.sceneGraphTypes import SceneGraphNode
from blenderproc.python import bp_python

parser = argparse.ArgumentParser()
parser.add_argument("--house_file", required=True, help="Path to house.json file")
parser.add_argument("--chair_obj", required=True, help="Path to chair object")
parser.add_argument("--output_dir", default=".", help="Output directory")
args = parser.parse_args()

blenderproc.init()

# Load objects from house.json file
label_mapping_file = os.path.join(os.path.dirname(args.house_file), "label_mapping.csv")
blenderproc.load_scene_from_json(args.house_file, label_mapping_file)

# Load chair object and replace all chair objects in the scene
chair_mat = bpy.data.materials.new(name="ChairMat")
chair_mat.use_nodes = True
bsdf_node = chair_mat.node_tree.nodes["Principled BSDF"]
bsdf_node.inputs['Base Color'].default_value = (0.5, 0.5, 0.5, 1.0)

chair_nodes = blenderproc.get_objects_by_label("chair")
for chair_node in chair_nodes:
    chair_node.materials = [chair_mat]
    floor_nodes = blenderproc.get_objects_by_label("floor")
    bp_python.replace_object(chair_node, args.chair_obj, ignore_collisions=floor_nodes, copy_properties=True)

# Filter out invalid objects and make Suncg objects emit light
scene_graph = blenderproc.get_scene_graph()
for node in scene_graph.children:
    if node.type == "MESH" and not node.valid:
        blenderproc.delete_object(node.name)
    elif node.type == "MESH" and "suncg" in node.name.lower():
        node.emission_strength = 1.0

# Initialize point sampler and BVH tree
sampler = blenderproc.PointSampler()
bvh_tree = blenderproc.BVHTree()

# Sample camera poses inside the house
obstacle_distance = 1.0
coverage_threshold = 0.4
sampler.sample_camera_poses(bvh_tree, obstacle_distance, coverage_threshold)

# Enable normal, depth, and segmentation rendering, add alpha channel to textures
blenderproc.enable_render_pass("NORMAL")
blenderproc.enable_render_pass("DEPTH")
blenderproc.enable_render_pass("SEGMENTATION")
blenderproc.enable_alpha_channel()

# Render the scene and write the rendered data to a .hdf5 file
output_path = os.path.join(args.output_dir, "output.hdf5")
blenderproc.render(output_path, save_format="HDF5")
```