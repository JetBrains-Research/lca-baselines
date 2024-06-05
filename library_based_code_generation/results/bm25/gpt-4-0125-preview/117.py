```python
import blenderproc as bproc
import argparse
import numpy as np

def main(obj_file_path, texture_dir, output_dir):
    # Initialize blenderproc
    bproc.init()

    # Load the scene from the .obj file
    bproc.loader.load_obj(obj_file_path, texture_dir)

    # Define a mapping for labeling objects
    object_label_mapping = {
        "wall": "Wall",
        "floor": "Floor",
        "ceiling": "Ceiling",
        "lamp": "Lamp",
        "ceiling_lamp": "CeilingLamp"
    }

    # Label objects based on the provided mapping
    for obj in bproc.object.get_all_mesh_objects():
        if obj.get_name() in object_label_mapping:
            obj.set_cp("category_id", object_label_mapping[obj.get_name()])

    # Separate walls, floors, and ceilings and assign labels
    for obj in bproc.object.get_all_mesh_objects():
        if "Wall" in obj.get_name():
            obj.set_cp("category_id", "Wall")
        elif "Floor" in obj.get_name():
            obj.set_cp("category_id", "Floor")
        elif "Ceiling" in obj.get_name():
            obj.set_cp("category_id", "Ceiling")

    # Make lamp and ceiling objects emit light
    for obj in bproc.object.get_all_mesh_objects():
        if "Lamp" in obj.get_name() or "CeilingLamp" in obj.get_name():
            bproc.types.Light(obj).set_energy(10)

    # Create a BVH tree containing all objects in the scene
    bproc.object.create_bvh_tree_multi_objects(bproc.object.get_all_mesh_objects())

    # Sample camera locations
    camera_positions = bproc.sampler.upper_hemisphere_surface_sampler(
        radius=1.5, location=[0, 0, 1], up_vector=[0, 0, 1], max_tries=100
    )

    for position, rotation in camera_positions:
        # Check for obstacles and scene coverage
        if bproc.camera.scene_coverage_score(position) > 0.5:
            bproc.camera.add_camera_pose(position, rotation)

    # Enable normal, depth, and segmentation rendering
    bproc.renderer.enable_normals_output()
    bproc.renderer.enable_depth_output()
    bproc.renderer.enable_segmentation_output()

    # Render the scene
    data = bproc.renderer.render()

    # Write data to a .hdf5 file
    bproc.writer.write_hdf5(output_dir, data)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load a 3D scene from an .obj file and texture files.")
    parser.add_argument("obj_file_path", help="The path to the .obj file")
    parser.add_argument("texture_dir", help="The directory containing texture files")
    parser.add_argument("output_dir", help="The directory to write the output .hdf5 file")
    args = parser.parse_args()

    main(args.obj_file_path, args.texture_dir, args.output_dir)
```