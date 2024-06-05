 ```python
import bpy
import os
import blenderproc

parser = argparse.ArgumentParser()
parser.add_argument("blend_file", help="Path to the blend file")
parser.add_argument("haven_dir", help="Path to the haven directory")
parser.add_argument("output_dir", help="Path to the output directory")
args = parser.parse_args()

blenderproc.init()
bpy.ops.wm.open_mainfile(filepath=args.blend_file)

hdri_files = [f for f in os.listdir(args.haven_dir) if f.endswith(".exr")]
hdri_file = os.path.join(args.haven_dir, random.choice(hdri_files))
bpy.context.scene.world.use_nodes = True
nodes = bpy.context.scene.world.node_tree.nodes
for node in nodes:
    if node.type == 'Background':
        nodes.remove(node)
bg_node = nodes.new(type='ShaderNodeTexEnvironment')
bg_node.environment_texture.image = bpy.data.images.load(hdri_file)

light = bpy.data.lights.new(name="Point Light", type='POINT')
light.data.energy = 10.0
light.location = (0, 10, 0)
bpy.context.scene.collection.objects.link(light)

poi = bpy.data.objects["Camera"]
poi_location = poi.location

camera_poses = blenderproc.sample_camera_poses(poi_location, 5)

for pose in camera_poses:
    bpy.context.view_layer.camera.location = pose
    bpy.ops.render.render(write_still=True, filepath=os.path.join(args.output_dir, f"frame_{pose[0]}.png"))

blenderproc.set_render_settings(normal_render=True, depth_render=True)
blenderproc.render(output_format='hdf5', output_path=os.path.join(args.output_dir, 'output.hdf5'))

blenderproc.cleanup()
```