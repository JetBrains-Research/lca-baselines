```python
import blenderproc as bproc
import argparse
import numpy as np
import random

def sample_poses(num_objects):
    poses = []
    for _ in range(num_objects):
        pos = np.random.uniform([-0.5, -0.5, 0], [0.5, 0.5, 1])
        rot = np.random.uniform([0, 0, 0], [360, 360, 360])
        poses.append(bproc.math.build_transformation_mat(pos, np.radians(rot)))
    return poses

parser = argparse.ArgumentParser()
parser.add_argument('bop_parent_dir', help="Path to the BOP datasets parent directory")
parser.add_argument('cc_textures_dir', help="Path to the CC textures directory")
parser.add_argument('output_dir', help="Path to the output directory")
parser.add_argument('num_scenes', type=int, help="Number of scenes to generate")
args = parser.parse_args()

bproc.init()

# Load BOP objects
itodd_objs = bproc.loader.load_bop('itodd', args.bop_parent_dir)
tless_objs = bproc.loader.load_bop('tless', args.bop_parent_dir)

# Load BOP dataset intrinsics, set shading and hide objects
for obj in itodd_objs + tless_objs:
    obj.set_shading_mode('auto')
    obj.hide(True)

# Create room
floor = bproc.object.create_primitive('PLANE', size=2)
ceiling = bproc.object.create_primitive('PLANE', size=2, location=[0, 0, 2])
walls = [bproc.object.create_primitive('PLANE', size=2, location=[i, 0, 1], rotation=[0, np.radians(90), 0]) for i in [-1, 1]]
walls += [bproc.object.create_primitive('PLANE', size=2, location=[0, i, 1], rotation=[np.radians(90), 0, 0]) for i in [-1, 1]]
room_planes = [floor, ceiling] + walls

# Enable rigidbody for room planes
for plane in room_planes:
    plane.enable_rigidbody(False, friction=0.5, restitution=0.5)

# Create light plane and point light
light_plane = bproc.object.create_primitive('PLANE', size=1, location=[0, 0, 1.9])
light_plane.set_name("LightPlane")
bproc.lighting.create_point_light(location=[0, 0, 1.8], energy=1000)

# Load CC textures
cc_textures = bproc.loader.load_ccmaterials(args.cc_textures_dir)

# Rendering settings
bproc.renderer.enable_depth_output(activate_antialiasing=False)
bproc.renderer.set_max_amount_of_samples(350)

for scene_id in range(args.num_scenes):
    # Sample BOP objects
    sampled_objs = random.sample(itodd_objs + tless_objs, k=10)
    
    # Randomize materials
    for obj in sampled_objs:
        obj.randomize_materials()
    
    # Set physics
    for obj in sampled_objs:
        obj.enable_rigidbody(True)
    
    # Sample two light sources
    bproc.lighting.light_surface([floor, ceiling], num_lights=2, energy=5)
    
    # Assign random CC texture to room planes
    for plane in room_planes:
        plane.replace_materials(random.choice(cc_textures))
    
    # Sample object poses
    poses = sample_poses(len(sampled_objs))
    for obj, pose in zip(sampled_objs, poses):
        obj.set_pose(pose)
    
    # Check collisions, simulate physics, and fix final poses
    bproc.object.simulate_physics_and_fix_final_poses(min_simulation_time=1, max_simulation_time=2, check_object_interval=1)
    
    # Create BVH tree for camera obstacle checks
    bproc.camera.create_bvh_tree_cam_obstacle_check()
    
    # Generate camera poses
    cam_poses = bproc.camera.sample_poses(cam2world_matrix=bproc.math.build_transformation_mat([0, 0, 1], [np.radians(45), 0, 0]),
                                          number_of_cam_poses=5, 
                                          min_distance_to_obstacle=0.3)
    
    for pose in cam_poses:
        bproc.camera.add_camera_pose(pose)
    
    # Render the pipeline
    data = bproc.renderer.render()
    bproc.writer.write_bop(args.output_dir, dataset='custom', depths=data['depth'], colors=data['colors'], color_file_format='JPEG', append_to_existing_output=True)
    
    # After each scene, disable rigidbody and hide objects
    for obj in sampled_objs:
        obj.disable_rigidbody()
        obj.hide(True)
```