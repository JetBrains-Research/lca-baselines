 ```python
import blenderproc
import argparse
import csv
import random
import os

def load_scene(scene_file):
    blenderproc.import_scene(scene_file)

def load_materials(material_file):
    blenderproc.import_materials(material_file)

def load_textures(materials):
    for material in materials:
        if blenderproc.get_material_texture_names(material):
            blenderproc.import_texture(blenderproc.get_material_texture_names(material)[0])

def label_objects(csv_file):
    with open(csv_file, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            obj = blenderproc.get_object_by_name(row['name'])
            if obj:
                blenderproc.set_object_category(obj, int(row['category']))

def assign_random_materials(objects, materials):
    num_objects = len(objects)
    num_materials = len(materials)
    num_objects_to_assign = int(num_objects * 0.4)
    selected_objects = random.sample(objects, num_objects_to_assign)
    for obj in selected_objects:
        material_index = random.randint(0, num_materials - 1)
        blenderproc.set_object_material(obj, materials[material_index])

def extract_floors_and_ceilings(objects):
    for obj in objects:
        if 'wall' in blenderproc.get_object_name(obj).lower():
            bbox = blenderproc.get_object_bounding_box(obj)
            floor_z = min(bbox[1][2], bbox[5][2])
            ceiling_z = max(bbox[1][2], bbox[5][2])
            if abs(floor_z - bbox[1][2]) < abs(ceiling_z - bbox[5][2]):
                blenderproc.set_object_category(obj, 1)  # floor
            else:
                blenderproc.set_object_category(obj, 2)  # ceiling

def enable_light(objects, category_id):
    for obj in objects:
        if blenderproc.get_object_category(obj) == category_id:
            blenderproc.set_object_emission(obj, [1.0, 1.0, 1.0, 1.0])

def create_bvh_tree(objects):
    blenderproc.create_bvh_tree(objects)

def sample_camera_locations(objects, num_samples):
    blenderproc.sample_camera_locations(objects, num_samples)

def render_scene(output_dir):
    blenderproc.enable_render_pass('normal')
    blenderproc.enable_render_pass('depth')
    blenderproc.enable_render_pass('segmentation')
    blenderproc.render(output_dir + '/scene.hdf5')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--scene', type=str, required=True)
    parser.add_argument('--material', type=str, required=True)
    parser.add_argument('--texture', type=str, required=True)
    parser.add_argument('--output', type=str, required=True)
    parser.add_argument('--csv', type=str, required=True)
    args = parser.parse_args()

    load_scene(args.scene)
    materials = load_materials(args.material)
    load_textures(materials)
    objects = blenderproc.get_objects_by_type('MESH')
    label_objects(args.csv)
    assign_random_materials(objects, materials)
    walls = [obj for obj in objects if 'wall' in blenderproc.get_object_name(obj).lower()]
    extract_floors_and_ceilings(walls)
    lamps = blenderproc.get_objects_by_type('LAMP')
    ceilings = [obj for obj in objects if blenderproc.get_object_category(obj) == 2]
    enable_light(lamps, 0)
    enable_light(ceilings, 2)
    objects_for_bvh = [obj for obj in objects if blenderproc.get_object_category(obj) not in [0, 3]]
    create_bvh_tree(objects_for_bvh)
    num_samples = 10
    sample_camera_locations(objects_for_bvh, num_samples)
    render_scene(args.output)
```