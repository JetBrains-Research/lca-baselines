 Here is the Python code based on your instructions:

```python
import os
import bpy
import blenderproc

def init_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("blend_file", help="Path to the blend file")
    parser.add_argument("haven_dir", help="Path to the haven directory")
    parser.add_argument("output_dir", help="Path to the output directory")
    return parser

def main(args):
    bpy.ops.wm.open_mainfile(filepath=args.blend_file)

    blenderproc.init()
    blenderproc.load_blend(args.blend_file)

    bkg_img_path = blenderproc.get_random_world_background_hdr_img_path_from_haven(args.haven_dir)
    bpy.context.scene.world.background_image = bkg_img_path

    light = bpy.data.objects["Point Light"]
    light.location = (0, 0, 5)
    light.energy = 10

    poi = min_and_max_point(bpy.context.scene.objects)
    camera_poses = sample_poses_around_point(poi)

    blenderproc.enable_normal_output()
    blenderproc.enable_depth_output()

    blenderproc.render_pipeline()

    output_file = os.path.join(args.output_dir, "output.hdf5")
    blenderproc.write_to_hdf_file(output_file)

def sample_poses_around_point(poi):
    # Implement your logic to sample five camera poses around the point of interest
    pass

if __name__ == "__main__":
    args = init_parser().parse_args()
    check_if_setup_utilities_are_at_the_top()
    main(args)
```

Please note that the `sample_poses_around_point` function is not implemented in the provided code. You will need to implement this function according to your specific requirements. Also, the `check_if_setup_utilities_are_at_the_top` function is a placeholder for a function that checks if the necessary setup utilities are at the top of the script. You can replace it with your own implementation or remove it if it's not necessary in your case.