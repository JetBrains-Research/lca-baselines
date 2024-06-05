```python
import blenderproc as bproc
import numpy as np

def main(blend_file_path, haven_dir_path, output_dir):
    # Initialize the parser
    parser = bproc.parser.Parser()
    parser.add_argument("blend_file_path", type=str, help="Path to the blend file.")
    parser.add_argument("haven_dir_path", type=str, help="Path to the HDRI haven directory.")
    parser.add_argument("output_dir", type=str, help="Path to the output directory.")
    args = parser.parse_args(['--blend_file_path', blend_file_path, '--haven_dir_path', haven_dir_path, '--output_dir', output_dir])

    # Initialize blenderproc
    bproc.init()

    # Load the blend file
    bproc.loader.load_blend(args.blend_file_path)

    # Set a random HDRI background
    hdri_paths = bproc.loader.glob_paths(f"{args.haven_dir_path}/*.hdr")
    bproc.loader.load_hdri_environment_map(np.random.choice(hdri_paths))

    # Define and set a point light
    light = bproc.types.Light()
    light.set_type("POINT")
    light.set_location([3, -3, 3])
    light.set_energy(1000)

    # Compute a point of interest and sample camera poses
    poi = bproc.object.compute_poi()
    camera_poses = bproc.camera.sample_poses_around_point(poi, number_of_samples=5)

    # Enable normal and depth rendering
    bproc.renderer.enable_normals_output()
    bproc.renderer.enable_depth_output()

    # Render the pipeline
    data = bproc.renderer.render()

    # Write the data to a .hdf5 container
    bproc.writer.write_hdf5(args.output_dir, data)

if __name__ == "__main__":
    blend_file_path = "/path/to/blend/file.blend"
    haven_dir_path = "/path/to/haven/directory"
    output_dir = "/path/to/output/directory"
    main(blend_file_path, haven_dir_path, output_dir)
```