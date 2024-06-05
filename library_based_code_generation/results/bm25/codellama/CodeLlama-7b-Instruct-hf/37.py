  Here is the Python code that performs the tasks you specified using the blenderproc library:
```
import blenderproc
import numpy as np

# Parse command line arguments
house_json_path = "path/to/house.json"
chair_obj_path = "path/to/chair.obj"
output_dir = "path/to/output/directory"

# Initialize blenderproc
blenderproc.init()

# Load objects from the house.json file into the scene using a label mapping from a csv file
label_mapping = from_csv("path/to/label_mapping.csv")
blenderproc.load_and_create(house_json_path, label_mapping)

# Load a chair object from the provided path and replace all chair objects in the scene with this chair object
blenderproc.load_and_create(chair_obj_path, "Chair")
blenderproc.replace_objects("Chair", "Chair")

# Filter out invalid objects from the scene
blenderproc.filter_objects(lambda obj: obj.type != "Invalid")

# Make all Suncg objects in the scene emit light
blenderproc.light_suncg_scene()

# Initialize a point sampler for sampling locations inside the loaded house and a bvh tree containing all mesh objects
point_sampler = blenderproc.PointSampler(blenderproc.get_all_mesh_objects())
bvh_tree = blenderproc.create_bvh_tree_multi_objects(point_sampler.get_points())

# Sample camera poses inside the house, ensuring that obstacles are at least 1 meter away from the camera and the view covers at least 40% of the scene
camera_poses = blenderproc.sample_camera_poses(bvh_tree, 1, 40)

# Enable normal, depth, and segmentation rendering
blenderproc.enable_normal_depth_segmentation()

# Render the scene and write the rendered data to a .hdf5 file in the specified output directory
blenderproc.render_scene(output_dir, "hdf5")
```
Note that this code assumes that the `house.json` file and the `chair.obj` file are located in the same directory as the Python script, and that the `label_mapping.csv` file is located in the same directory as the `house.json` file. You will need to modify the file paths and other parameters as needed to match your specific use case.