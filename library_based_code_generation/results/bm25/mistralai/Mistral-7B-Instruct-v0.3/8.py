 Here's a Python script that follows your instructions using the BlenderProc library. Please note that this script assumes you have the necessary BlenderProc utilities installed and imported.

```python
import argparse
import os
import csv
import bpy
import numpy as np
from blenderproc.pipeline.utils import check_if_setup_utilities_are_at_the_top

def load_and_label_objects(scene_file, csv_file):
    bpy.ops.wm.open_mainfile(filepath=scene_file)
    check_if_setup_utilities_are_at_the_top()

    objects = get_all_blender_mesh_objects()
    with open(csv_file, 'r') as f:
        reader = csv.reader(f)
        next(reader)  # skip header
        for row in reader:
            obj_name, label = row
            for obj in objects:
                if obj.name == obj_name:
                    obj.name = label

def load_materials_and_assign(objects):
    materials = load_and_create('materials.blend')
    assigned_materials = set()

    for material in materials:
        if len([obj for obj in objects if get_type_and_value_from_mat(material, 'name') == obj.material_slots[0].name]) > 0:
            assigned_materials.add(material)

    for obj in objects:
        if obj.material_slots[0].name in assigned_materials:
            obj.material_slots[0].material = materials[obj.material_slots[0].name]

def load_textures(materials):
    for material in materials.values():
        if len([obj for obj in get_all_mesh_objects() if material in obj.material_slots]) > 0:
            bpy.ops.image.open(filepath='textures/' + get_type_and_value_from_mat(material, 'texture'))
            image = bpy.data.images[bpy.context.active_object.name]
            material.use_nodes = True
            texture_node = material.node_tree.nodes['Material Output'].inputs[0]
            texture_node.links[0].from_socket = image.nodes['Image Texture']

def extract_floor_ceiling(objects):
    floors = []
    ceilings = []

    for obj in objects:
        if 'Floor' in obj.name:
            floors.append(obj)
        elif 'Ceiling' in obj.name:
            ceilings.append(obj)

    _assign_materials_to_floor_wall_ceiling(floors, 'floor')
    _assign_materials_to_floor_wall_ceiling(ceilings, 'ceiling')

def light_scene(objects):
    for obj in objects:
        if 'Lamp' in obj.name:
            obj.data.use_shadow = False
            obj.data.energy = 1000
        elif 'Ceiling' in obj.name:
            obj.data.use_shadow = False
            obj.data.energy = 100

def create_bvh_tree(objects):
    bvh_tree = create_bvh_tree_multi_objects(objects)
    sample_locations = bvh_tree.sample_locations(100)

    for location in sample_locations:
        ray = bvh_tree.ray_from_location(location)
        if not any([ray.intersects(obj.bound_box) for obj in objects if obj.type == 'MESH']):
            location.z += 2
            if scene_coverage_score(objects, location) > 0.5:
                break

def main(scene_file, texture_file, material_file, csv_file, output_dir):
    load_and_label_objects(scene_file, csv_file)
    objects = get_all_mesh_objects()

    validate_and_standardizes_configured_list(objects, 40)
    load_materials_and_assign(objects)
    load_textures(bpy.data.materials)

    extract_floor_ceiling(objects)
    light_scene(objects)

    bpy.ops.object.select_all(action='DESELECT')
    for obj in objects:
        obj.select_set(True)

    bpy.context.view_layer.objects.active = objects[0]
    bpy.ops.object.duplicate_move(OT_type='OBJECT', move_type='TRANSLATION', value=(0, 0, 1))

    create_bvh_tree(objects)

    bpy.ops.render.render(write_still=True, filepath=f'{output_dir}/render.png')
    _colorize_objects_for_instance_segmentation(objects)

    bpy.ops.wm.save_as_mainfile(filepath=f'{output_dir}/scene.blend')

    bpy.ops.export_scene.hdf5(filepath=f'{output_dir}/data.hdf5', use_selection=True, use_normal=True, use_depth=True, use_segmentation=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", required=True)
    parser.add_argument("--texture", required=True)
    parser.add_argument("--material", required=True)
    parser.add_argument("--csv", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    check_if_setup_utilities_are_at_the_top()

    create_mesh_objects_from_file(args.scene)
    move_and_duplicate_furniture(args.scene)

    main(args.scene, args.texture, args.material, args.csv, args.output)
```

This script assumes that you have the necessary BlenderProc utilities installed and imported. It loads a scene, labels its objects based on a CSV file, loads materials, assigns them to objects, loads textures, extracts floors and ceilings, lights the scene, creates a BVH tree, renders the scene, and saves the data to an HDF5 file.