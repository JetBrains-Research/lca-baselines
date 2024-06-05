  Here is the Python code that initializes a scene in Blender using the blenderproc library:
```
import blenderproc
import argparse
import os

# Parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--scene", required=True, help="Path to scene file")
parser.add_argument("--texture", required=True, help="Path to texture file")
parser.add_argument("--material", required=True, help="Path to material file")
parser.add_argument("--output", required=True, help="Output directory")
args = parser.parse_args()

# Load scene and label objects
blenderproc.load_and_create(args.scene)
blenderproc.get_node_connected_to_the_output_and_unlink_it()
blenderproc.get_the_one_node_with_type("Object")

# Load materials and randomly assign them to 40% of objects
blenderproc.load_and_create(args.material)
blenderproc.check_if_setup_utilities_are_at_the_top()
blenderproc.assign_materials_to_floor_wall_ceiling()

# Load textures for materials
blenderproc.load_and_create(args.texture)

# Extract floors and ceilings from wall objects
blenderproc.transform_and_colorize_object()
blenderproc.create_mesh_objects_from_file()
blenderproc.move_and_duplicate_furniture()

# Assign category IDs to floors and ceilings
blenderproc.min_and_max_point()
blenderproc.calc_vec_and_normals()
blenderproc.perform_and_condition_check()

# Make all lamp objects emit light and make all ceiling objects emit a bit of light
blenderproc.light_suncg_scene()

# Create BVH tree containing all mesh objects
blenderproc.create_bvh_tree_multi_objects()

# Sample camera locations above the floor
blenderproc.scene_coverage_score()
blenderproc.get_all_mesh_objects()
blenderproc.get_all_blender_mesh_objects()

# Enable normal, depth, and segmentation rendering
blenderproc.enable_normal_depth_segmentation_rendering()

# Render the scene and write data to .hdf5 file
blenderproc.render_scene()
blenderproc.write_data_to_hdf5_file(args.output)
```
Note that this code assumes that the `blenderproc` library is installed and that the `blender` executable is in the system's `PATH`. Additionally, this code assumes that the scene, texture, and material files are in the same directory as the Python script.