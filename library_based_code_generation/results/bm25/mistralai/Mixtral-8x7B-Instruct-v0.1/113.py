 import bpy
from blenderproc.public import *

parser = Parser(blend_file_path="path/to/blend/file", haven_dir_path="path/to/haven/directory", output_dir_path="path/to/output/directory")
init_blenderproc(parser)
load_blend(parser.blend_file_path)

hdri_path = get_random_world_background_hdr_img_path_from_haven(parser.haven_dir_path)
set_background_hdri(hdri_path)

point_light = define_point_light(location=(0, 0, 10), energy=1000)

point_of_interest = min_and_max_point(get_type_and_value_from_mat("Cube", "position"))
camera_poses = sample_camera_poses_around_point(point_of_interest, num_poses=5)

set_normal_output(True)
set_depth_output(True)
render_pipeline(parser)

hdf5_file_path = f"{parser.output_dir_path}/output.hdf5"
write_to_hdf_file(hdf5_file_path, "rendered_images", "rgb", get_image_as_numpy_array(parser.output_name))
write_to_hdf_file(hdf5_file_path, "rendered_images", "depth", get_depth_as_numpy_array(parser.output_name))

hide_links_and_collision_inertial_objs()
cleanup_blenderproc()