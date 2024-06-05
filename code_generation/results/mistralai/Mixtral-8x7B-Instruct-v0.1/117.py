 ```python
import argparse
import os
import bpy
import blenderproc
from blenderproc.publicTypes.objectLabel import ObjectLabel
from blenderproc.publicTypes.cameraSample import CameraSample

def load_scene(obj_file, texture_dir, object_mapping):
    bpy.ops.import_scene.obj(filepath=obj_file)
    for obj in bpy.context.scene.objects:
        if obj.name in object_mapping:
            obj.data.materials[0].use_nodes = True
            tree = obj.data.materials[0].node_tree
            links = tree.links
            nodes = tree.nodes
            diffuse_node = nodes["Principled BSDF"]
            tex_image_node = nodes.new("ShaderNodeTexImage")
            tex_image_node.location = (-300, 0)
            tex_image_node.image = bpy.data.images.load(os.path.join(texture_dir, object_mapping[obj.name]))
            links.new(diffuse_node.inputs['Base Color'], tex_image_node.outputs['Color'])
            obj.blenderproc_object_label = object_mapping[obj.name]

    walls = [obj for obj in bpy.context.scene.objects if obj.blenderproc_object_label == 'wall']
    floor = [obj for obj in bpy.context.scene.objects if obj.blenderproc_object_label == 'floor'][0]
    ceiling = [obj for obj in bpy.context.scene.objects if obj.blenderproc_object_label == 'ceiling'][0]
    lamp = [obj for obj in bpy.context.scene.objects if obj.blenderproc_object_label == 'lamp'][0]

    for wall in walls:
        wall.blenderproc_object_label = 'wall'
    floor.blenderproc_object_label = 'floor'
    ceiling.blenderproc_object_label = 'ceiling'
    lamp.blenderproc_object_label = 'lamp'

    bpy.ops.object.select_all(action='DESELECT')
    for obj in walls + [floor, ceiling, lamp]:
        obj.select_set(True)
    bpy.ops.object.join()

    bvh_tree = bpy.data.objects.new("BVH_Tree", None)
    bvh_tree.blenderproc_create_bvh_tree = True
    bpy.context.scene.collection.objects.link(bvh_tree)

def sample_camera_poses(num_samples, floor_height):
    min_distance_from_floor = 2
    min_coverage_score = 0.7
    samples = []
    for _ in range(num_samples):
        loc_x = (2 * bpy.context.scene.camera.location.x - bpy.context.scene.camera.data.type) * 5
        loc_y = bpy.context.scene.camera.location.y
        loc_z = bpy.context.scene.camera.location.z + (5 + min_distance_from_floor) * (1 + 0.2 * (bpy.context.scene.frame_current - 1) / bpy.context.scene.frame_end)
        rot_x = bpy.context.scene.camera.rotation_euler.x
        rot_y = bpy.context.scene.camera.rotation_euler.y
        rot_z = bpy.context.scene.camera.rotation_euler.z
        obj = bpy.data.objects.new("Camera", bpy.context.scene.camera)
        obj.location = (loc_x, loc_y, loc_z)
        obj.rotation_euler = (rot_x, rot_y, rot_z)
        bpy.context.scene.collection.objects.link(obj)
        coverage_score = blenderproc.publicFunctions.calculate_scene_coverage_score(obj, floor_height)
        if coverage_score > min_coverage_score:
            samples.append(CameraSample(obj.location, obj.rotation_euler))
            bpy.context.scene.collection.objects.unlink(obj)
    return samples

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("obj_file", help="Path to the .obj file")
    parser.add_argument("texture_dir", help="Path to the directory containing texture files")
    parser.add_argument("output_dir", help="Path to the output directory")
    parser.add_argument("--object_mapping", help="Mapping of object names to labels")
    args = parser.parse_args()

    blenderproc.initialize()
    blenderproc.start_blender()

    if args.object_mapping:
        object_mapping = {k.strip(): v.strip() for k, v in [x.split(':') for x in args.object_mapping.split(',')]}
    else:
        object_mapping = {'Cabinet': 'wall', 'Chair': 'wall', 'Cushion': 'wall', 'Table': 'wall', 'Television': 'wall', 'Vase': 'wall', 'Window': 'wall', 'Door': 'wall', 'Floor': 'floor', 'Ceiling': 'ceiling', 'Lamp': 'lamp'}

    floor_height = 0
    load_scene(args.obj_file, args.texture_dir, object_mapping)

    for obj in bpy.context.scene.objects:
        if obj.blenderproc_object_label == 'floor':
            floor_height = obj.location.z
            obj.data.materials[0].use_nodes = True
            tree = obj.data.materials[0].node_tree
            links = tree.links
            nodes = tree.nodes
            diffuse_node = nodes["Principled BSDF"]
            diffuse_node.inputs['Base Color'].default_value = (0.8, 0.8, 0.8, 1)
            links.new(diffuse_node.inputs['Base Color'], diffuse_node.outputs['Base Color'])

        if obj.blenderproc_object_label == 'ceiling':
            obj.data.materials[0].use_nodes = True
            tree = obj.data.materials[0].node_tree
            links = tree.links
            nodes = tree.nodes
            diffuse_node = nodes["Principled BSDF"]
            diffuse_node.inputs['Emission'].default_value = (0.8, 0.8, 0.8, 1)
            links.new(diffuse_node.inputs['Emission'], diffuse_node.outputs['Emission'])

        if obj.blenderproc_object_label == 'lamp':
            obj.data.type = 'POINT'
            obj.data.energy = 1000
            obj.data.distance = 10
            obj.data.shadow_soft_size = 0.1

    samples = sample_camera_poses(10, floor_height)

    blenderproc.publicFunctions.set_render_settings(samples=samples,
                                                    output_path=os.path.join(args.output_dir, "render.hdf5"),
                                                    use_normal_map=True,
                                                    use_depth_map=True,
                                                    use_segmentation_map=True)

    blenderproc.render()

    blenderproc.terminate()

if __name__ == "__main__":
    main()
```