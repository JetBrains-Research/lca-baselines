import blenderproc
import argparse

# Parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--scene_file", type=str, help="Path to scene file")
parser.add_argument("--texture_file", type=str, help="Path to texture file")
parser.add_argument("--material_file", type=str, help="Path to material file")
parser.add_argument("--output_dir", type=str, help="Output directory")
args = parser.parse_args()

# Load scene
scene = blenderproc.load_scene(args.scene_file)

# Label objects based on mapping from CSV file
blenderproc.label_objects(scene, "mapping.csv")

# Load materials and randomly assign to 40% of objects
blenderproc.load_materials(scene, args.material_file)
blenderproc.randomly_assign_materials(scene, 0.4)

# Load textures for materials assigned to objects
blenderproc.load_textures(scene, args.texture_file)

# Extract floors and ceilings from wall objects
blenderproc.extract_floors_and_ceilings(scene)

# Assign appropriate category IDs to floors and ceilings
blenderproc.assign_category_ids(scene)

# Make lamp objects emit light
blenderproc.make_lamps_emit_light(scene)

# Make ceiling objects emit a bit of light
blenderproc.make_ceilings_emit_light(scene)

# Create BVH tree containing all mesh objects
bvh_tree = blenderproc.create_bvh_tree(scene)

# Sample camera locations above the floor
blenderproc.sample_camera_locations(scene)

# Ensure no obstacles in front of camera and scene coverage score is not too low
blenderproc.ensure_no_obstacles(scene)
blenderproc.check_scene_coverage(scene)

# Enable normal, depth, and segmentation rendering
blenderproc.enable_rendering(scene, ["normal", "depth", "segmentation"])

# Render scene and write data to .hdf5 file
blenderproc.render_scene(scene)
blenderproc.write_data_to_hdf5(scene, args.output_dir)