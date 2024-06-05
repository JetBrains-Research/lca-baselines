```python
import argparse
import blenderproc as bproc
import numpy as np

def parse_arguments():
    parser = argparse.ArgumentParser(description='Process input paths.')
    parser.add_argument('house_json', type=str, help='Path to the house.json file')
    parser.add_argument('chair_path', type=str, help='Path to the chair object')
    parser.add_argument('--output_dir', type=str, default='output', help='Optional output directory')
    return parser.parse_args()

def main():
    args = parse_arguments()

    # Initialize blenderproc
    bproc.init()

    # Load house objects with label mapping
    label_mapping = bproc.utility.LabelIdMapping.from_csv("label_mapping.csv")
    house_objects = bproc.loader.load_json(args.house_json, label_mapping)

    # Load the new chair object
    new_chair = bproc.object.create_mesh_objects_from_file(args.chair_path)

    # Replace all chair objects in the scene
    for obj in bproc.object.get_all_mesh_objects():
        if obj.get_cp("category_id") == label_mapping.get_id("chair"):
            bproc.object.replace_objects(obj, new_chair, copy_properties=True, collision_ignore={"floor"}, random_rotation={'z': (0, 360)})

    # Filter out invalid objects
    bproc.object.filter_invalid_objects()

    # Make Suncg objects emit light
    bproc.light.light_suncg_scene()

    # Initialize point sampler and bvh tree
    point_sampler = bproc.sampler.PointSampler()
    bvh_tree = bproc.object.create_bvh_tree_multi_objects(bproc.object.get_all_blender_mesh_objects())

    # Sample camera poses
    camera_poses = point_sampler.sample_inside_hull(house_objects, min_dist=1.0, min_view_coverage=0.4)
    for pose in camera_poses:
        bproc.camera.add_camera_pose(pose)

    # Enable rendering settings
    bproc.renderer.enable_normals_output()
    bproc.renderer.enable_depth_output()
    bproc.renderer.enable_segmentation_output()
    bproc.renderer.add_alpha_channel_to_textures()

    # Render the scene
    data = bproc.renderer.render()

    # Write rendered data to .hdf5 file
    bproc.writer.write_hdf5(args.output_dir, data)

if __name__ == "__main__":
    main()
```