```python
import blenderproc

parser = blenderproc.initialize_parser(bop_datasets_parent_dir='path_to_parent_dir', cc_textures='path_to_cc_textures', output_dir='path_to_output_dir', num_scenes=5)
blenderproc.init_library(parser)

blenderproc.load_bop_objects(parser, datasets=['itodd', 'tless'])
blenderproc.load_bop_intrinsics(parser)
blenderproc.set_shading_and_hide_objects(parser)

blenderproc.create_room_with_primitive_planes(parser)
blenderproc.enable_rigidbody_for_planes(parser)
blenderproc.create_light_plane_and_point_light(parser)

blenderproc.load_cc_textures(parser)
blenderproc.define_function_to_sample_poses(parser)

blenderproc.enable_depth_rendering_without_antialiasing(parser)
blenderproc.set_max_samples_for_color_rendering(parser)

for scene in range(parser.num_scenes):
    blenderproc.sample_bop_objects(parser)
    blenderproc.randomize_materials(parser)
    blenderproc.set_physics(parser)
    blenderproc.sample_two_light_sources(parser)
    blenderproc.assign_random_cc_texture_to_room_planes(parser)
    blenderproc.sample_object_poses(parser)
    blenderproc.check_collisions(parser)
    blenderproc.simulate_physics_and_fix_final_poses(parser)

    blenderproc.create_BVH_tree_for_camera_obstacle_checks(parser)
    blenderproc.generate_camera_poses_with_obstacle_check(parser, min_distance=0.3)

    blenderproc.render_pipeline_and_write_data_in_bop_format(parser)

    blenderproc.disable_rigidbody_and_hide_objects(parser)

```