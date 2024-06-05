```python
import argparse
import random
import csv
import bpy
import blenderproc as bproc

def load_materials_and_textures(material_path, texture_path):
    materials = bproc.loader.load_materials(material_path)
    for material in materials:
        if random.random() < 0.4:
            bproc.material.set_principled_shader_value(material, "Base Color", value=[random.random() for _ in range(3)] + [1])
            texture = bproc.loader.load_texture(texture_path)
            bproc.material.set_principled_shader_value(material, "Base Color Texture", texture)
    return materials

def label_objects_from_csv(scene_path, csv_path):
    object_label_mapping = {}
    with open(csv_path, mode='r') as infile:
        reader = csv.reader(infile)
        object_label_mapping = {rows[0]:rows[1] for rows in reader}
    
    for obj in bproc.loader.load_blend(scene_path):
        if obj.get_name() in object_label_mapping:
            obj.set_cp("category_id", object_label_mapping[obj.get_name()])

def main(scene_path, material_path, texture_path, csv_path, output_dir):
    parser = argparse.ArgumentParser(description='Initialize a scene in Blender using blenderproc.')
    parser.add_argument('--scene_path', type=str, help='Path to the scene file', default=scene_path)
    parser.add_argument('--material_path', type=str, help='Path to the material file', default=material_path)
    parser.add_argument('--texture_path', type=str, help='Path to the texture file', default=texture_path)
    parser.add_argument('--csv_path', type=str, help='Path to the CSV file for object labeling', default=csv_path)
    parser.add_argument('--output_dir', type=str, help='Directory to save the output', default=output_dir)
    args = parser.parse_args()

    bproc.init()

    # Load the scene
    label_objects_from_csv(args.scene_path, args.csv_path)

    # Load and assign materials and textures
    materials = load_materials_and_textures(args.material_path, args.texture_path)

    # Process objects in the scene
    for obj in bproc.object.get_all_mesh_objects():
        if "Wall" in obj.get_name():
            bproc.object.create_primitive("PLANE", size=2).set_cp("category_id", "floor")
            bproc.object.create_primitive("PLANE", size=2).set_cp("category_id", "ceiling")
        elif "Lamp" in obj.get_name():
            obj.set_cp("emit_light", True)
        elif "Ceiling" in obj.get_name():
            obj.set_cp("emit_light", True)

    # Create BVH tree and sample camera locations
    bproc.camera.sample_locations_above_mesh(min_distance_to_mesh=0.5, max_distance_to_mesh=1.5, number_of_samples=10)

    # Enable rendering of different passes
    bproc.renderer.enable_normals_output()
    bproc.renderer.enable_depth_output()
    bproc.renderer.enable_segmentation_output()

    # Render the scene
    data = bproc.renderer.render()

    # Write output
    bproc.writer.write_hdf5(args.output_dir, data)

if __name__ == "__main__":
    main("path/to/scene.blend", "path/to/materials", "path/to/textures", "path/to/labels.csv", "path/to/output")
```