  Here is the Python code based on the instruction:
```
import blenderproc
import os

# Initialize the parser with the path to the blend file, the path to the haven directory, and the output directory
parser = blenderproc.Parser(blend_file_path, haven_dir_path, output_dir_path)

# Initialize the blenderproc library and load the blend file into the scene
blenderproc.init_library()
blenderproc.load_blend(blend_file_path)

# Set a random HDRi from the haven directory as the background
background_path = blenderproc.get_random_world_background_hdr_img_path_from_haven(haven_dir_path)
blenderproc.set_background(background_path)

# Define a point light and set its location and energy level
light = blenderproc.PointLight()
light.location = (0, 0, 0)
light.energy = 10
blenderproc.add_object(light)

# Compute a point of interest and sample five camera poses around it
point_of_interest = blenderproc.get_min_and_max_point()
camera_poses = blenderproc.sample_camera_poses(point_of_interest, 5)

# Enable normal and depth rendering
blenderproc.enable_normal_rendering()
blenderproc.enable_depth_rendering()

# Render the pipeline and write the data to a .hdf5 container in the specified output directory
blenderproc.render_pipeline(camera_poses)
blenderproc.write_to_hdf_file(output_dir_path)
```
Note that this code assumes that the `blenderproc` library is installed and that the `blend_file_path`, `haven_dir_path`, and `output_dir_path` variables are defined and contain the appropriate file paths.