  import blenderproc
import os
import random

# Parse command line arguments
house_json_path = "path/to/house.json"
chair_obj_path = "path/to/chair.obj"
output_dir = "path/to/output/dir"

# Initialize blenderproc
blenderproc.init()

# Load objects from house.json file
label_mapping = blenderproc.load_label_mapping("path/to/label_mapping.csv")
house_objects = blenderproc.load_house(house_json_path, label_mapping)

# Load chair object and replace all chair objects in scene
chair_object = blenderproc.load_object(chair_obj_path)
chair_object.pose.rotation.z = random.uniform(0, 360)
blenderproc.replace_objects(house_objects, chair_object, ignore_collisions=True, copy_properties=True)

# Filter out invalid objects
valid_objects = blenderproc.filter_objects(house_objects, lambda obj: obj.is_valid)

# Make all Suncg objects in scene emit light
blenderproc.set_lighting(valid_objects, "suncg")

# Initialize point sampler and bvh tree
point_sampler = blenderproc.PointSampler(valid_objects)
bvh_tree = blenderproc.BVHTree(valid_objects)

# Sample camera poses inside house
camera_poses = blenderproc.sample_camera_poses(point_sampler, bvh_tree, min_distance=1, min_coverage=0.4)

# Enable normal, depth, and segmentation rendering
blenderproc.enable_rendering(valid_objects, "normal", "depth", "segmentation")

# Add alpha channel to textures
blenderproc.add_alpha_channel(valid_objects)

# Render scene and write to .hdf5 file
output_path = os.path.join(output_dir, "output.hdf5")
blenderproc.render_scene(valid_objects, camera_poses, output_path)