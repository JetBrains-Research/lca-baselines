 Here is a Python script that uses the BlenderProc library to perform the tasks as described:

```python
import argparse
import os
import bpy
import numpy as np
from blenderproc import BlenderProc
from blenderproc.utilities import setup_utilities
from blenderproc.utilities.geometry_utilities import calc_vec_and_normals
from blenderproc.utilities.object_utilities import transform_and_colorize_object, hide_links_and_collision_inertial_objs
from blenderproc.utilities.scene_utilities import min_and_max_point, scene_coverage_score
from blenderproc.utilities.lighting_utilities import light_suncg_scene
from blenderproc.utilities.bvh_utilities import create_bvh_tree_multi_objects
from blenderproc.utilities.camera_utilities import move_and_duplicate_furniture, perform_and_condition_check

def load_and_label_scene(obj_mapping, scene_file, texture_files):
    bpy.ops.object.select_all(action='DESELECT')
    bpy.ops.import_scene.obj(filepath=scene_file, use_materials=True, use_textures=True, use_smooth_normals=True)

    for obj_name, label in obj_mapping.items():
        obj = bpy.data.objects[obj_name]
        obj.name = label

    return bpy.data.objects

def separate_walls_floors_ceilings(objects):
    walls = []
    floors = []
    ceilings = []

    for obj in objects:
        if 'Wall' in obj.name:
            walls.append(obj)
        elif 'Floor' in obj.name:
            floors.append(obj)
        elif 'Ceiling' in obj.name:
            ceilings.append(obj)

    return walls, floors, ceilings

def make_objects_emit_light(objects):
    for obj in objects:
        if 'Lamp' in obj.name:
            obj.data.emissive_strength = 10.0
        if 'Ceiling' in obj.name:
            light_suncg_scene(obj)

def create_bvh_tree(objects):
    create_bvh_tree_multi_objects(objects)

def sample_cameras(bvh_tree, output_dir):
    camera_locations = []
    camera_rotations = []

    for i in range(10):
        camera_location, camera_rotation = move_and_duplicate_furniture(bvh_tree.root, 2.0, 1.5, 0.0)
        camera_location[1] += 1.0
        camera_rotations.append(np.array([0.0, 0.0, 0.0]))

        if not perform_and_condition_check(bvh_tree, camera_location, camera_rotation, 0.5, 0.5, 0.5):
            camera_locations.pop()
            camera_rotations.pop()

    for i, (camera_location, camera_rotation) in enumerate(zip(camera_locations, camera_rotations)):
        camera_name = f'Camera_{i}'
        bpy.ops.object.camera_add(location=camera_location, rotation=camera_rotation)
        camera = bpy.data.objects[camera_name]
        camera.name = camera_name
        camera.data.type = 'PERSP'
        camera.data.lens = 50.0
        camera.data.angle_x = 0.0
        camera.data.angle_y = 0.0
        camera.data.angle_z = 0.0

        output_file = os.path.join(output_dir, f'{camera_name}.hdf5')
        bpy.context.scene.render.filepath = output_file
        bpy.ops.render.render(write_still=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", required=True, help="Path to the .obj scene file")
    parser.add_argument("--textures", nargs='+', help="Paths to the texture files")
    parser.add_argument("--output", required=True, help="Output directory for the rendered images")
    args = parser.parse_args()

    # Initialize BlenderProc
    bp = BlenderProc()
    setup_utilities.check_if_setup_utilities_are_at_the_top()

    # Load the scene and texture files
    scene_file = args.scene
    texture_files = args.textures
    obj_mapping = {'Model_001': 'Wall', 'Model_002': 'Wall', 'Model_003': 'Wall', 'Model_004': 'Floor', 'Model_005': 'Ceiling', 'Model_006': 'Lamp', 'Model_007': 'Ceiling'}
    objects = load_and_label_scene(obj_mapping, scene_file, texture_files)

    # Separate walls, floors, and ceilings
    walls, floors, ceilings = separate_walls_floors_ceilings(objects)

    # Make lamp and ceiling objects emit light
    make_objects_emit_light(objects)

    # Create BVH tree
    bvh_tree = create_bvh_tree(objects)

    # Hide links and collision inertial objects
    hide_links_and_collision_inertial_objs(objects)

    # Validate and standardize the configured list
    bp.validate_and_standardizes_configured_list(objects)

    # Simulate physics and fix final poses
    bp.simulate_physics_and_fix_final_poses(objects)

    # Create mesh objects from file for furniture
    bp.create_mesh_objects_from_file('_SceneNetLoader', objects)

    # Calculate normals and vertices for all objects
    bp.calc_vec_and_normals(objects)

    # Transform and colorize objects
    bp.transform_and_colorize_object(objects)

    # Set up lighting
    bp.light_suncg_scene()

    # Create BVH tree
    bp.create_bvh_tree_multi_objects(objects)

    # Sample cameras
    sample_cameras(bvh_tree, args.output)
```

This script assumes that you have already installed the BlenderProc library and set up your environment correctly. Make sure to replace the `obj_mapping` variable with the appropriate object names and labels for your specific scene. Also, adjust the `sample_cameras` function as needed to meet your specific requirements for camera placement, scene coverage score, and obstacle avoidance.