```python
import blenderproc as bproc
import argparse
import numpy as np
import random

parser = argparse.ArgumentParser()
parser.add_argument('bop_parent_dir', help="Path to the BOP datasets parent directory")
parser.add_argument('cc_textures', help="Path to the CC textures directory")
parser.add_argument('output_dir', help="Path to the output directory")
parser.add_argument('num_scenes', type=int, help="Number of scenes to generate")
args = parser.parse_args()

bproc.init()

# Load BOP objects
bproc.loader.load_bop(bop_path=args.bop_parent_dir,
                      dataset_name='itodd',
                      mm2m=True)
bproc.loader.load_bop(bop_path=args.bop_parent_dir,
                      dataset_name='tless',
                      mm2m=True)

# Load BOP dataset intrinsics
bproc.camera.load_intrinsics_from_bop(bop_path=args.bop_parent_dir)

# Create room
floor = bproc.object.create_primitive('PLANE', scale=[2, 2, 1])
walls = [bproc.object.create_primitive('PLANE', scale=[2, 2, 1]) for _ in range(4)]
for i, wall in enumerate(walls):
    wall.set_location([0, 0, 1])
    wall.set_rotation_euler([np.pi/2 * (i % 2), 0, np.pi/2 * ((i + 1) // 2)])

# Enable rigidbody for room planes
for plane in [floor] + walls:
    plane.enable_rigidbody(False, collision_shape='MESH')

# Create light plane and point light
light_plane = bproc.object.create_primitive('PLANE', scale=[5, 5, 1], location=[0, 0, 3])
light_plane.set_name("LightPlane")
bproc.lighting.add_light_emission_to_object(light_plane, emission_strength=10)
bproc.object.create_point_light(location=[0, 0, 4], energy=1000)

# Load CC textures
cc_textures = bproc.loader.load_ccmaterials(args.cc_textures)

def sample_6dof_poses(num_objects):
    poses = []
    for _ in range(num_objects):
        pos = np.random.uniform([-0.5, -0.5, 0], [0.5, 0.5, 2])
        rot = np.random.uniform([0, 0, 0], [np.pi, np.pi, np.pi])
        poses.append((pos, rot))
    return poses

# Enable depth rendering
bproc.renderer.enable_depth_output(activate_antialiasing=False)

# Set maximum samples for color rendering
bproc.renderer.set_max_amount_of_samples(350)

for _ in range(args.num_scenes):
    # Sample BOP objects
    objects_to_sample = random.sample(bproc.object.get_all_mesh_objects(), 5)
    
    # Randomize materials
    bproc.material.random_sample_materials_for_each_obj(objects_to_sample, cc_textures)
    
    # Set physics
    for obj in objects_to_sample:
        obj.enable_rigidbody(True, collision_shape='CONVEX_HULL')
    
    # Sample light sources
    bproc.lighting.light_surface_sampling(light_plane, num_lights=2)
    
    # Assign random CC texture to room planes
    for plane in [floor] + walls:
        bproc.material.sample_materials_for_objects([plane], cc_textures)
    
    # Sample object poses
    poses = sample_6dof_poses(len(objects_to_sample))
    for obj, (pos, rot) in zip(objects_to_sample, poses):
        obj.set_location(pos)
        obj.set_rotation_euler(rot)
    
    # Check collisions and simulate physics
    bproc.object.simulate_physics_and_fix_final_poses(min_simulation_time=1, max_simulation_time=2)
    
    # Create BVH tree for camera obstacle checks
    bvh_tree = bproc.object.create_bvh_tree_multi_objects(objects_to_sample)
    
    # Generate camera poses
    camera_poses = bproc.camera.sample_poses_around_object(floor,
                                                           number_of_samples=5,
                                                           distance_range=[1, 2],
                                                           azimuth_range=[0, 360],
                                                           elevation_range=[-45, 45])
    for pose in camera_poses:
        if bvh_tree.ray_test(pose, floor.get_location())[0] > 0.3:
            bproc.camera.add_camera_pose(pose)
    
    # Render the pipeline
    bproc.renderer.render_pipeline_writer(args.output_dir, append_to_existing_output=True)
    
    # Disable rigidbody and hide objects for next scene
    for obj in objects_to_sample:
        obj.disable_rigidbody()
        obj.hide()
```