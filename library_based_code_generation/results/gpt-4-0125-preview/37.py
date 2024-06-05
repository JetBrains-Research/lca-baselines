```python
import argparse
import blenderproc as bproc
import numpy as np

def main():
    parser = argparse.ArgumentParser(description='Process some paths.')
    parser.add_argument('house_json', type=str, help='Path to the house.json file')
    parser.add_argument('chair_path', type=str, help='Path to the chair object')
    parser.add_argument('--output_dir', type=str, default='output', help='Optional output directory')
    args = parser.parse_args()

    bproc.init()

    # Load the house
    label_mapping = bproc.utility.LabelIdMapping(csv_filename="label_mapping.csv")
    objects = bproc.loader.load_json(args.house_json, label_mapping=label_mapping)

    # Load the new chair object
    new_chair = bproc.loader.load_obj(args.chair_path)[0]

    # Replace all chair objects in the scene
    for obj in objects:
        if obj.get_cp("category_id") == label_mapping.get_id("chair"):
            new_obj = new_chair.copy()
            new_obj.set_location(obj.get_location())
            new_obj.set_scale(obj.get_scale())
            z_rotation = np.random.uniform(0, 360)
            new_obj.set_rotation_euler([0, 0, np.deg2rad(z_rotation)])
            bproc.object.replace_object(obj, new_obj, copy_properties=True, ignore_collision_with=["floor"])

    # Filter out invalid objects
    bproc.object.filter_invalid_objects()

    # Make Suncg objects emit light
    for obj in bproc.object.get_scene_objects():
        if "Suncg" in obj.get_name():
            obj.make_emissive()

    # Initialize point sampler and bvh tree
    point_sampler = bproc.sampler.PointSampler()
    bvh_tree = bproc.object.create_bvh_tree_from_objects(bproc.object.get_scene_objects())

    # Sample camera poses
    camera_poses = point_sampler.sample_inside_hull(bproc.object.get_scene_objects(), min_dist=1.0, min_view_percentage=0.4)
    for pose in camera_poses:
        bproc.camera.add_camera_pose(pose)

    # Enable rendering of normal, depth, and segmentation
    bproc.renderer.enable_normals_output()
    bproc.renderer.enable_depth_output()
    bproc.renderer.enable_segmentation_output()
    bproc.renderer.add_alpha_channel_to_textures()

    # Render the scene
    data = bproc.renderer.render()
    bproc.writer.write_hdf5(args.output_dir, data)

if __name__ == "__main__":
    main()
```