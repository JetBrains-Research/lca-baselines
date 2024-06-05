import sys
import blenderproc

# Parse command line arguments
house_json_path = sys.argv[1]
chair_obj_path = sys.argv[2]
output_dir = sys.argv[3] if len(sys.argv) > 3 else None

# Initialize blenderproc
bp = blenderproc.BlenderProc()

# Load objects from house.json file with label mapping
bp.load_objects_from_json(house_json_path, label_mapping_csv)

# Load chair object and replace chair objects in scene
bp.load_chair_object(chair_obj_path, ignore_collisions=True, copy_properties=True, random_rotation=True)

# Filter out invalid objects
bp.filter_invalid_objects()

# Make all Suncg objects emit light
bp.make_suncg_objects_emit_light()

# Initialize point sampler and bvh tree
bp.initialize_point_sampler()
bp.initialize_bvh_tree()

# Sample camera poses inside house
bp.sample_camera_poses(obstacle_distance=1, view_coverage=0.4)

# Enable rendering settings
bp.enable_rendering_settings(normal=True, depth=True, segmentation=True, alpha_channel=True)

# Render scene and write to .hdf5 file
bp.render_scene(output_dir)