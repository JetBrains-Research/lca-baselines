 import argparse
import os
import random
import numpy as np
from blenderproc.public import (
    create_mesh_objects_from_file,
    replace_objects,
    get_all_mesh_objects,
    merge_objects,
    visible_objects,
    _colorize_objects_for_instance_segmentation,
    load_and_create,
    transform_and_colorize_object,
    get_all_blender_mesh_objects,
    get_type_and_value_from_mat,
    join_with_other_objects,
    create_bvh_tree_multi_objects,
    set_camera_parameters_from_config_file,
    load_bop_scene,
    min_and_max_point,
    from_csv,
    light_suncg_scene
)

parser = argparse.ArgumentParser()
parser.add_argument("--house_file", required=True, help="Path to house.json file")
parser.add_argument("--chair_path", required=True, help="Path to chair object")
parser.add_argument("--output_dir", default=None, help="Optional output directory")
args = parser.parse_args()

if args.output_dir is None:
    args.output_dir = os.getcwd()

# Initialize blenderproc
from blenderproc.modules import ModuleManager
ModuleManager.initialize()

# Load objects from house.json file into the scene using a label mapping from a csv file
house_objects, label_mapping = load_bop_scene(args.house_file)

# Load chair object from the provided path and replace all chair objects in the scene with this chair object
chair_objects = create_mesh_objects_from_file(args.chair_path)
replace_objects(house_objects, chair_objects, ignore_collisions=["floor"], copy_properties=True, randomize_rotation_z=True)

# Filter out invalid objects from the scene
scene_objects = get_all_blender_mesh_objects()
scene_objects = [obj for obj in scene_objects if obj.name not in ["__empty__"]]

# Make all Suncg objects in the scene emit light
for obj in scene_objects:
    if "suncg" in obj.name:
        light_suncg_scene(obj)

# Initialize a point sampler for sampling locations inside the loaded house and a bvh tree containing all mesh objects
sampler_points, bvh_tree = from_csv("path/to/sampler_points.csv"), create_bvh_tree_multi_objects(scene_objects)

# Sample camera poses inside the house, ensuring that obstacles are at least 1 meter away from the camera and the view covers at least 40% of the scene
min_point, max_point = min_and_max_point(bvh_tree)
camera_poses = []
for _ in range(100):
    sample_point = np.random.uniform(min_point, max_point, size=3)
    if np.linalg.norm(sample_point - min_point) > 1 or np.linalg.norm(sample_point - max_point) > 1:
        camera_poses.append(sample_point)

# Add these camera poses to the scene
for pose in camera_poses:
    set_camera_parameters_from_config_file(pose, "path/to/camera_config.json")

# Enable normal, depth, and segmentation rendering. Add an alpha channel to textures
ModuleManager.execute_module("enable_normal_depth_segmentation_rendering")
ModuleManager.execute_module("add_alpha_channel_to_textures")

# Render the scene and write the rendered data to a .hdf5 file in the specified output directory
ModuleManager.execute_module("render_scene", output_path=os.path.join(args.output_dir, "output.hdf5"))
ModuleManager.cleanup()