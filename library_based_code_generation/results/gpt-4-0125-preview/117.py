```python
import sys
import blenderproc as bproc
import numpy as np

def main(obj_file_path, texture_dir, output_dir):
    # Initialize the blenderproc library
    bproc.init()

    # Load the 3D scene from the specified .obj file and texture files
    bproc.loader.load_obj(obj_file_path, texture_dir)

    # Define a mapping for labeling objects
    label_mapping = {
        "wall": "Wall",
        "floor": "Floor",
        "ceiling": "Ceiling",
        "lamp": "Lamp",
        "ceiling_lamp": "CeilingLamp"
    }

    # Label objects based on the provided mapping
    for obj in bproc.object.get_all_mesh_objects():
        if obj.get_name() in label_mapping:
            obj.set_cp("category_id", label_mapping[obj.get_name()])

    # Separate walls, floors, and ceilings into distinct objects and assign labels
    bproc.object.split_by_material()

    # Make lamp and ceiling objects emit light
    for obj in bproc.object.get_all_mesh_objects():
        if obj.get_cp("category_id") in ["Lamp", "CeilingLamp"]:
            bproc.lighting.add_emission_to_material(obj)

    # Create a BVH tree containing all objects in the scene
    bproc.object.create_bvh_tree()

    # Sample camera locations and rotations
    camera_positions = bproc.sampler.upper_hemisphere_surface(
        number_of_samples=10,
        radius=1.5,
        location=[0, 0, 1],
        surface_sample_area=0.5
    )

    for position, rotation in camera_positions:
        # Check for obstacles and scene coverage
        if not bproc.camera.scene_coverage_score(position) < 0.5 and not bproc.camera.obstacle_in_view(position):
            bproc.camera.add_camera_pose(position, rotation)

    # Enable normal, depth, and segmentation rendering
    bproc.renderer.enable_normals_output()
    bproc.renderer.enable_depth_output()
    bproc.renderer.enable_segmentation_output()

    # Render the scene
    data = bproc.renderer.render()

    # Write the data to a .hdf5 file in the specified output directory
    bproc.writer.write_hdf5(output_dir, data)

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python script.py <obj_file_path> <texture_dir> <output_dir>")
        sys.exit(1)

    main(sys.argv[1], sys.argv[2], sys.argv[3])
```