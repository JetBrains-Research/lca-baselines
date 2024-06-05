 Here is the Python code that follows your instructions using the BlenderProc library:

```python
import argparse
import blenderbim
import numpy as np
import bpy
import bmesh
from blenderproc.pipeline.utils import check_if_setup_utilities_are_at_the_top

def load_and_replace_chair(obj_path, label_csv):
    chair = bpy.data.objects[obj_path]
    original_chairs = [obj for obj in bpy.data.objects if obj.type == 'MESH' and obj.name.startswith('chair_')]

    for chair_obj in original_chairs:
        if chair_obj != chair:
            label = label_csv.get(chair_obj.name)
            new_chair = chair.copy()
            new_chair.name = f'chair_{label}'
            new_chair.location = chair_obj.location
            new_chair.rotation_euler = (np.random.rand() * 2 * np.pi, 0, 0)
            new_chair.select_set(True)
            bpy.context.view_layer.objects.active = new_chair
            bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
            new_chair.select_set(False)
            bpy.ops.object.select_all(action='DESELECT')
            new_chair.select_set(True)
            bpy.ops.object.replace_keep_transform(objects={chair_obj: new_chair})

def filter_invalid_objects(scene):
    for obj in scene.objects:
        if not obj.type == 'MESH':
            obj.select_set(True)
    bpy.ops.object.delete()

def setup_scene(house_json, chair_obj_path, label_csv, output_dir=None):
    check_if_setup_utilities_are_at_the_top()

    bpy.ops.wm.read_homefile()
    bpy.ops.import_scene.bim('INVOKE_DEFAULT', filepath=house_json)
    blenderbim.utils.load_csv(label_csv)

    load_and_replace_chair(chair_obj_path, label_csv)
    filter_invalid_objects(bpy.context.scene)

def setup_rendering(scene):
    bpy.context.scene.cycles.samples = 100
    bpy.context.scene.render.engine = 'CYCLES'
    bpy.context.scene.render.filepath = f'{output_dir}/render.hdf5'
    bpy.context.scene.render.image_settings.color_depth = '16'
    bpy.context.scene.render.image_settings.file_format = 'OPENEXR'
    bpy.context.scene.render.alpha_method = 'DIRECT'
    bpy.context.scene.cycles.use_denoising = True

def main(args):
    setup_scene(args.house_json, args.chair_obj_path, args.label_csv, args.output_dir)
    setup_rendering(bpy.context.scene)

    # 6. Make all Suncg objects in the scene emit light.
    # This step is not explicitly mentioned in the instructions, but it seems like it's intended.
    # Assuming that Suncg objects are labeled as 'suncg_*.*'.
    suncg_objects = [obj for obj in bpy.data.objects if obj.name.startswith('suncg_')]
    for obj in suncg_objects:
        obj.data.use_nodes = True
        materials_node = obj.data.materials[0].node_tree
        emission_node = materials_node.nodes['Emission']
        emission_node.inputs['Strength'].default_value = 10

    # 7. Initialize a point sampler for sampling locations inside the loaded house and a bvh tree containing all mesh objects.
    # These steps are not explicitly mentioned in the instructions, but they seem necessary for the rest of the pipeline.
    bvh_tree = create_bvh_tree_multi_objects(get_all_blender_mesh_objects(bpy.context.scene))
    point_sampler = blenderproc.samplers.PointSampler(bvh_tree, min_samples=1000, max_samples=10000)

    # 8. Sample camera poses inside the house, ensuring that obstacles are at least 1 meter away from the camera and the view covers at least 40% of the scene. Add these camera poses to the scene.
    camera_poses = point_sampler.sample_cameras(min_distance=1, view_fraction=0.4)
    for pose in camera_poses:
        cam = bpy.data.cameras.new('Camera')
        obj = bpy.data.objects.new('Camera', cam)
        obj.location = pose['location']
        obj.rotation_euler = pose['rotation']
        bpy.context.scene.collection.objects.link(obj)

    # 9. Enable normal, depth, and segmentation rendering. Add an alpha channel to textures.
    bpy.context.scene.render.layers[0].use_pass_normal = True
    bpy.context.scene.render.layers[0].use_pass_depth = True
    bpy.context.scene.render.layers[0].use_pass_z_buffer = True
    bpy.context.scene.render.layers[0].use_pass_alpha = True

    # 10. Render the scene and write the rendered data to a .hdf5 file in the specified output directory.
    bpy.ops.render.render(write_still=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--house_json", type=str, required=True)
    parser.add_argument("--chair_obj_path", type=str, required=True)
    parser.add_argument("--label_csv", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default=".")
    args = parser.parse_args()
    main(args)
```

This code assumes that you have the BlenderProc library installed and that the house.json file, chair object, and label_csv file are in the same directory as the script. The output .hdf5 file will be saved in the specified output directory.