 Here is a simplified Python script that follows your instructions. Please note that this script assumes you have the necessary BlenderProc dependencies installed and the correct file paths.

```python
import argparse
import bpy
import csv
import numpy as np
import blenderproc

def load_csv(file_path):
    with open(file_path, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        object_mapping = {row['name']: row['category'] for row in reader}
    return object_mapping

def assign_materials(scene, material_file, object_mapping):
    materials = blenderproc.io.load_materials(material_file)
    for obj in scene.objects:
        if obj.type == 'MESH':
            if obj.name in object_mapping:
                obj.data.materials = [materials[material_name] for material_name in np.random.choice(list(materials.keys()), len(obj.data.materials) // 2, replace=False)]

def load_textures(scene, materials):
    for material in materials.values():
        if material.texture_slots:
            material.texture_slots[0].image = bpy.data.images.load(material.texture_slots[0].image.name)

def extract_floor_ceiling(scene):
    floor_category = 'floor'
    ceiling_category = 'ceiling'

    floor_objects = [obj for obj in scene.objects if obj.name.endswith('_floor') and obj.type == 'MESH']
    ceiling_objects = [obj for obj in scene.objects if obj.name.endswith('_ceiling') and obj.type == 'MESH']

    for obj in floor_objects:
        obj.category = bpy.context.object.categories.find(floor_category)
    for obj in ceiling_objects:
        obj.category = bpy.context.object.categories.find(ceiling_category)

def enable_lighting(scene):
    for lamp in scene.objects:
        if lamp.type == 'LAMP':
            lamp.data.energy = 1000
    for obj in scene.objects:
        if obj.type == 'MESH' and obj.category == bpy.context.object.categories.find('ceiling'):
            obj.data.emissive_intensity = 0.1

def create_bvh_tree(scene):
    bvh_tree = blenderproc.io.create_bvh_tree(scene, camera_height=2.0, min_camera_distance=1.0, min_scene_coverage=0.8)
    return bvh_tree

def render_and_save(bvh_tree, output_dir):
    renderer = blenderproc.Renderer(bvh_tree, 'NORMAL', 'DEPTH', 'SEGMENTATION')
    renderer.render(output_dir)
    renderer.save_data('hdf5', output_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", type=str, required=True)
    parser.add_argument("--texture", type=str, required=True)
    parser.add_argument("--material", type=str, required=True)
    parser.add_argument("--object_mapping", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()

    bpy.ops.wm.open_mainfile(filepath=args.scene)
    scene = bpy.context.scene

    object_mapping = load_csv(args.object_mapping)
    assign_materials(scene, args.material, object_mapping)
    load_textures(scene, blenderproc.io.load_materials(args.material))
    extract_floor_ceiling(scene)
    enable_lighting(scene)

    bvh_tree = create_bvh_tree(scene)
    render_and_save(bvh_tree, args.output)
```

This script loads a Blender scene, assigns materials to objects based on a CSV file, loads textures for the materials, extracts floors and ceilings, enables lighting, creates a BVH tree, renders the scene, and saves the data to an .hdf5 file.