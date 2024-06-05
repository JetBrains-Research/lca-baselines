import sys
import blenderproc

house_json_path = sys.argv[1]
chair_obj_path = sys.argv[2]
output_dir = sys.argv[3] if len(sys.argv) > 3 else None

blenderproc.init()

label_mapping_csv = "label_mapping.csv"
blenderproc.load_and_create(house_json_path, label_mapping_csv)

chair_obj = blenderproc.create_mesh_objects_from_file(chair_obj_path)
blenderproc.replace_objects(chair_obj, ignore_collisions=["floor"], copy_properties=True, random_rotation=True)

blenderproc.filter_invalid_objects()
blenderproc._colorize_objects_for_instance_segmentation()

blenderproc.light_suncg_scene()

point_sampler = blenderproc.create_bvh_tree_multi_objects()
bvh_tree = blenderproc.get_all_blender_mesh_objects()
camera_poses = blenderproc.load_bop_scene(point_sampler, bvh_tree, min_distance=1, min_view_coverage=0.4)
blenderproc.set_camera_parameters_from_config_file()

blenderproc.enable_rendering_modes(["normal", "depth", "segmentation"], add_alpha_channel=True)
blenderproc.render_scene(output_dir, output_format="hdf5")