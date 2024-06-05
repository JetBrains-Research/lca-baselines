import blenderproc

parser = argparse.ArgumentParser(description='BOP dataset generation')
parser.add_argument('--bop_parent_dir', type=str, help='Path to the BOP datasets parent directory')
parser.add_argument('--cc_textures', type=str, help='Path to the CC textures directory')
parser.add_argument('--output_dir', type=str, help='Path to the output directory')
parser.add_argument('--num_scenes', type=int, help='Number of scenes to generate')
args = parser.parse_args()

blenderproc.init()

blenderproc.load_bop_objects('itodd')
blenderproc.load_bop_objects('tless')

blenderproc.create_room()
blenderproc.enable_rigidbody()

blenderproc.create_light_plane()
blenderproc.create_point_light()

blenderproc.load_cc_textures(args.cc_textures)
blenderproc.define_pose_sampling_function()

blenderproc.enable_depth_rendering(antialiasing=False)
blenderproc.set_max_color_samples(100)

for scene in range(args.num_scenes):
    blenderproc.sample_bop_objects()
    blenderproc.randomize_materials()
    blenderproc.set_physics()
    blenderproc.sample_light_sources(2)
    blenderproc.assign_random_cc_texture_to_room_planes()
    blenderproc.sample_object_poses()
    blenderproc.check_collisions()
    blenderproc.simulate_physics()
    blenderproc.fix_final_poses()

    blenderproc.create_bvh_tree_for_camera_obstacle_check()
    blenderproc.generate_camera_poses(0.3)

    blenderproc.render_pipeline()
    blenderproc.write_data_in_bop_format()

    blenderproc.disable_rigidbody()
    blenderproc.hide_objects()