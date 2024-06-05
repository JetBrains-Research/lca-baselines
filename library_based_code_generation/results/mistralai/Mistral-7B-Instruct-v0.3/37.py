 Here is the Python code that follows your instructions using the BlenderProc library:

```python
import argparse
import blenderbim
import bpy
import numpy as np
import open3d as o3d

parser = argparse.ArgumentParser()
parser.add_argument('--house', type=str, required=True)
parser.add_argument('--chair', type=str, required=True)
parser.add_argument('--output', type=str, default='output')
args = parser.parse_args()

blenderbim.register()
bpy.ops.wm.blenderbim_open(filepath=args.house)

label_map = pd.read_csv('label_map.csv')
blenderbim.utils.load_objects_from_json(label_map, bpy.context.scene)

chair_obj = bpy.data.objects[args.chair]

def replace_chair(obj):
    if obj.type == 'MESH' and 'Chair' in obj.name:
        bpy.ops.object.select_all(action='DESELECT')
        obj.select_set(True)
        bpy.context.view_layer.objects.active = obj
        bpy.ops.object.duplicate_move(linked=True)
        new_chair = bpy.context.selected_objects[0]
        new_chair.location.z += 1
        new_chair.rotation_euler.z = np.random.uniform(0, 2 * np.pi)
        new_chair.copy_location_from(obj)
        new_chair.copy_rotation_quaternion_from(obj)
        obj.select_set(False)
        return new_chair
    return None

for obj in bpy.data.objects:
    replacement = replace_chair(obj)
    if replacement:
        obj.select_set(False)
        bpy.data.objects.remove(obj)

invalid_objects = [obj for obj in bpy.data.objects if obj.type != 'MESH']
for obj in invalid_objects:
    bpy.data.objects.remove(obj)

for obj in bpy.data.objects:
    if isinstance(obj, bpy.types.SuncgObject):
        obj.data.use_shadow = True
        obj.data.use_cast_shadow = True

bvh_tree = o3d.geometry.TriangleMesh.create_from_blender(bpy.context.scene)
point_sampler = o3d.utility.PointSampler(bvh_tree, 1000)

camera_poses = []
for point in point_sampler.sample():
    ray = o3d.utility.RayCastResult()
    if o3d.utility.ray_triangle_intersection(point, bvh_tree.get_bounding_box(), ray):
        if ray.get_distance() > 1:
            camera = o3d.camera.PinholeCameraIntrinsic(width=1024, height=768, fx=500, fy=500)
            camera.convert_from_intrinsic(ray.get_camera_pose())
            camera.compute_view_matrix_new()
            camera.compute_projection_matrix()
            camera_poses.append(camera.get_view_matrix())

bpy.context.scene.render.filepath = f'{args.output}/render.hdf5'
bpy.context.scene.cycles.samples = 100
bpy.ops.render.render(write_still=True)
```

This code assumes that you have a CSV file named `label_map.csv` that maps object names to labels. The CSV file should have two columns: the first column contains the object names in the Blender scene, and the second column contains the corresponding labels.

Also, this code uses Open3D for point sampling and ray casting. Make sure to install Open3D before running the script:

```bash
pip install open3d
```