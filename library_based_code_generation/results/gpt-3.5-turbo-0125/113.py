import blenderproc

blend_file_path = "path/to/blend/file.blend"
haven_directory_path = "path/to/haven/directory"
output_directory = "path/to/output/directory"

parser = blenderproc.Parser(blend_file_path, haven_directory_path, output_directory)
blenderproc.init()

blenderproc.load_blend_file()

blenderproc.set_random_hdri_from_directory()

blenderproc.define_point_light(location=(0, 0, 5), energy=100)

blenderproc.compute_point_of_interest()

blenderproc.sample_camera_poses(num_poses=5)

blenderproc.enable_normal_rendering()
blenderproc.enable_depth_rendering()

blenderproc.render_pipeline()

blenderproc.write_data_to_hdf5_container()