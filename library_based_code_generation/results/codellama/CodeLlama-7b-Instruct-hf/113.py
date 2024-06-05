  from blenderproc import BlenderProc
import os
import random

# Initialize the parser with the three arguments
parser = BlenderProc.Parser(
    blend_file_path='path/to/blend/file.blend',
    haven_dir_path='path/to/haven/directory',
    output_dir_path='path/to/output/directory'
)

# Initialize the blenderproc library and load the blend file into the scene
blenderproc = BlenderProc.BlenderProc(parser)
blenderproc.load_blend_file()

# Set a random hdri from the haven directory as the background
random_hdri = random.choice(os.listdir(parser.haven_dir_path))
blenderproc.set_background(os.path.join(parser.haven_dir_path, random_hdri))

# Define a point light and set its location and energy level
point_light = blenderproc.add_point_light()
point_light.location = (0, 0, 0)
point_light.energy = 10

# Compute a point of interest and sample five camera poses around it
point_of_interest = blenderproc.compute_point_of_interest()
camera_poses = blenderproc.sample_camera_poses(point_of_interest, 5)

# Enable normal and depth rendering
blenderproc.enable_normal_rendering()
blenderproc.enable_depth_rendering()

# Render the pipeline and write the data to a .hdf5 container in the specified output directory
blenderproc.render_pipeline(camera_poses)
blenderproc.write_hdf5_container(os.path.join(parser.output_dir_path, 'output.hdf5'))