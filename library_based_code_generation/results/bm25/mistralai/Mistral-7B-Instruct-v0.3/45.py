 Here is the Python code based on your instructions using the BlenderProc library:

```python
import bpy
import numpy as np
from blenderproc.frontend import BlenderProc

def initialize_parser():
    parser = BlenderProc.get_parser()
    parser.add_argument('--bop_parent_dir', type=str, required=True)
    parser.add_argument('--cc_textures', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--num_scenes', type=int, default=1)

    return parser

def setup_scene():
    bpy.ops.object.select_all(action='DESELECT')
    bpy.ops.mesh.primitive_plane_add(size=10, enter_editmode=False, location=(0, 0, 0))
    room_planes = bpy.context.selected_objects
    for plane in room_planes:
        plane.rigid_body.mass = 1.0
        plane.name = 'room_plane'

    light_plane = bpy.ops.mesh.primitive_plane_add(size=10, enter_editmode=False, location=(0, 0, 5))
    light_plane.name = 'light_plane'

    point_light = bpy.data.lights.new(name="Point Light", type='POINT')
    point_light.location = (0, 0, 5)

    # Load CC textures
    for texture in bpy.data.images.load(cc_textures, check_existing=True):
        texture.use_nodes = True
        bpy.context.scene.materials.new(name=texture.name)
        mat = bpy.context.active_object.material_slots[0].material
        mat.use_nodes = True
        mat.node_tree.nodes['Principled BSDF'].inputs[0].default_value = texture

def sample_poses(num_poses):
    poses = []
    for _ in range(num_poses):
        pose = np.random.uniform(-np.pi, np.pi, 3)
        quat = np.quaternion_from_euler(pose)
        poses.append(quat)
    return poses

def main():
    args = initialize_parser().parse_args()

    BlenderProc.initialize()
    bpy.ops.object.select_all(action='DESELECT')

    # Load BOP datasets and intrinsics
    bop_scenes = load_bop_scene(args.bop_parent_dir, ['itodd', 'tless'])
    bop_intrinsics = load_bop_intrinsics(args.bop_parent_dir)

    # Set shading and hide objects
    check_if_setup_utilities_are_at_the_top()
    hide_links_and_collision_inertial_objs()

    # Create BVH tree for camera obstacle checks
    bvh_tree = create_bvh_tree()

    for scene_index in range(args.num_scenes):
        bpy.context.scene = bop_scenes[scene_index]

        # Sample bop objects, randomize materials, set physics, sample two light sources, assign a random cc texture to room planes
        bop_objects = sample_bop_objects(bop_scenes[scene_index])
        random_sample_materials_for_each_obj(bop_objects)
        set_physics(bop_objects)
        sample_light_sources(2)
        assign_random_cc_texture_to_room_planes()

        # Sample object poses, check collisions, simulate physics and fix final poses
        object_poses = sample_poses(len(bop_objects))
        check_collisions(bop_objects, object_poses)
        simulate_physics_and_fix_final_poses(bop_objects, object_poses)

        # Render the pipeline and write data in bop format
        render_pipeline(bvh_tree)
        write_bop(args.output_dir, bop_scenes[scene_index], bop_intrinsics)

        # Disable rigidbody and hide objects
        for obj in bop_objects:
            obj.rigid_body.enabled = False
            obj.hide_viewport = True

    BlenderProc.finalize()

if __name__ == "__main__":
    main()
```

Please note that this code is a simplified version of the actual BlenderProc pipeline and assumes that the helper functions like `load_bop_scene`, `load_bop_intrinsics`, `sample_bop_objects`, `set_physics`, `sample_light_sources`, `assign_random_cc_texture_to_room_planes`, `sample_poses`, `check_collisions`, `simulate_physics_and_fix_final_poses`, `render_pipeline`, and `write_bop` are already defined. You may need to implement these functions based on the BlenderProc library documentation.