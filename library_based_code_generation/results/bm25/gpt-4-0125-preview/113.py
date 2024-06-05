```python
import blenderproc as bproc
import numpy as np

def main(blend_file_path, haven_directory, output_directory):
    # Initialize the blenderproc
    bproc.init()

    # Load the blend file
    bproc.loader.load_blend(blend_file_path)

    # Set a random HDRI background
    hdri_path = bproc.lighting.get_random_world_background_hdr_img_path_from_haven(haven_directory)
    bproc.lighting.load_hdr_img_as_world_background(hdri_path)

    # Define and set a point light
    point_light = bproc.types.Light()
    point_light.set_type("POINT")
    point_light.set_location([0, 0, 3])
    point_light.set_energy(1000)

    # Compute a point of interest
    poi = bproc.object.compute_poi()

    # Sample five camera poses around the point of interest
    camera_poses = bproc.camera.sample_poses_around_point(poi, number_of_samples=5)

    # Enable normal and depth rendering
    bproc.renderer.enable_normals_output()
    bproc.renderer.enable_depth_output()

    # Render the pipeline
    data = bproc.renderer.render()

    # Write the data to a .hdf5 container
    bproc.writer.write_to_hdf5(output_directory + "/output.hdf5", data)

if __name__ == "__main__":
    blend_file_path = "/path/to/blend/file.blend"
    haven_directory = "/path/to/haven/directory"
    output_directory = "/path/to/output/directory"
    main(blend_file_path, haven_directory, output_directory)
```