 ```python
import os
import random
import numpy as np
from blenderproc.geometry.primitives import create_plane
from blenderproc.materials import create_material
from blenderproc.physics import enable_rigidbody
from blenderproc.textures import load_texture
from blenderproc.utils.math import sample_6dof_pose
from blenderproc.utils.render import enable_depth_render, set_max_samples, render_pipeline
from blenderproc.utils.scene import create_camera, create_light, disable_rigidbody, hide_objects
from blenderproc.utils.geometry import create_bvh_tree

# Initialize blenderproc
import bpy
bpy.ops.bp_utils.init_bp()

# Set paths and number of scenes
bop_datasets_parent_dir = "/path/to/bop/datasets"
cc_textures_dir = "/path/to/cc/textures"
output_dir = "/path/to/output/directory"
num_scenes = 10

# Load BOP datasets
bpy.ops.bp_datasets.load_bop_dataset(dataset_name="itodd", datasets_parent_dir=bop_datasets_parent_dir)
bpy.ops.bp_datasets.load_bop_dataset(dataset_name="tless", datasets_parent_dir=bop_datasets_parent_dir)

# Load BOP dataset intrinsics
bpy.ops.bp_datasets.load_bop_intrinsics(datasets_parent_dir=bop_datasets_parent_dir)

# Set shading and hide objects
bpy.context.scene.render.engine = 'CYCLES'
bpy.context.scene.cycles.samples = 64
bpy.context.scene.view_layer.light_direct_enabled = False
bpy.context.scene.view_layer.light_indirect_enabled = False
hide_objects()

# Create room using primitive planes and enable rigidbody
room_planes = [
    create_plane(size=5, location=(2.5, 2.5, 0)),
    create_plane(size=5, location=(-2.5, 2.5, 0)),
    create_plane(size=5, location=(-2.5, -2.5, 0)),
    create_plane(size=5, location=(2.5, -2.5, 0))
]
enable_rigidbody(objects=room_planes)

# Create a light plane
light_plane = create_plane(size=10, location=(0, 0, 5))
create_material(object=light_plane, material_name="white_emission", emission_strength=10)

# Create a point light
point_light = create_light(type="POINT", energy=1000, location=(0, 0, 10))

# Load cc textures
load_texture(texture_file=os.path.join(cc_textures_dir, "color", "000000.png"), texture_name="white_texture")

# Define a function to sample 6-DoF poses
def sample_poses(num_objects):
    poses = []
    for _ in range(num_objects):
        pose = sample_6dof_pose(max_translation=2.5, max_rotation=0.5)
        poses.append(pose)
    return poses

# Sample bop objects, randomize materials, set physics, sample light sources, assign textures, and check collisions
for scene in range(num_scenes):
    # Sample bop objects
    bop_objects = bpy.data.objects.keys()
    random.shuffle(bop_objects)
    bop_objects = bop_objects[:len(room_planes)]

    # Randomize materials
    materials = [create_material(object=obj, material_name="white_texture") for obj in bop_objects]
    for i, mat in enumerate(materials):
        mat.node_tree.nodes["Principled BSDF"].base_color = (
            np.random.rand(3) * 0.5 + 0.5
        )

    # Set physics
    for obj, mat in zip(bop_objects, materials):
        obj.data.materials.append(mat)
        enable_rigidbody(objects=[obj])

    # Sample light sources
    light_source_1 = create_light(type="POINT", energy=500, location=(random.uniform(-1.5, 1.5), random.uniform(-1.5, 1.5), 3))
    light_source_2 = create_light(type="POINT", energy=500, location=(random.uniform(-1.5, 1.5), random.uniform(-1.5, 1.5), 7))

    # Assign a random cc texture to room planes
    room_plane_materials = [create_material(object=plane, material_name="white_texture") for plane in room_planes]
    for i, mat in enumerate(room_plane_materials):
        mat.node_tree.nodes["Principled BSDF"].base_color = (
            np.random.rand(3) * 0.5 + 0.5
        )
        room_planes[i].data.materials.append(mat)

    # Sample object poses
    bop_poses = sample_poses(len(bop_objects))

    # Check collisions
    for obj, pose in zip(bop_objects, bop_poses):
        obj.location = pose[:3]
        obj.rotation_euler = pose[3:]
        bpy.ops.object.modifier_add(type='BEVEL')
        bpy.ops.rigidbody.object_tool_add()
        bpy.context.object.rigid_body.collision_margin = 0.01
        bpy.context.object.modifiers["Bevel"].width = 0.01

    # Simulate physics and fix final poses
    bpy.ops.ptcache.bake_all(bake=True)

    # Create a BVH tree for camera obstacle checks
    bvh_tree = create_bvh_tree(objects=bop_objects)

    # Generate camera poses
    for i, obj in enumerate(bop_objects):
        cam_loc = obj.location + np.array([0, 0, 3])
        create_camera(location=cam_loc, rotation=(0, 0, 0))
        cam_obj = bpy.data.objects["Camera"]
        create_bvh_tree(objects=[cam_obj], name=f"camera_{i}")
        bpy.ops.object.select_all(action='DESELECT')
        cam_obj.select_set(True)
        bpy.context.view_layer.objects.active = cam_obj
        bpy.ops.object.parent_set(type='OBJECT', keep_transform=True)

        # Ensure obstacles are at least 0.3 meter away from the camera
        cam_bvh_tree = create_bvh_tree(objects=[cam_obj])
        while True:
            dists = bvh_tree.ray_cast(cam_bvh_tree.bound_box_center, cam_bvh_tree.bound_box_center - cam_loc)
            if dists[0][1] > 0.3:
                break
            cam_loc += np.random.uniform(-0.1, 0.1, 3)

        # Set camera parameters
        cam_obj.data.lens = 35
        cam_obj.data.type = 'PERSP'

    # Render the pipeline and write data in bop format
    enable_depth_render()
    set_max_samples(color_samples=128)
    render_pipeline(
        output_dir=output_dir,
        filename=f"scene_{scene:03d}",
        image_format="OPEN_EXR",
        data_format="BOP",
        data_format_version="1.1",
        camera_poses=[obj.matrix_world for obj in bpy.data.objects if obj.name.startswith("Camera")],
        intrinsics=bpy.context.scene.camera.data.lens,
        depth_image_path=