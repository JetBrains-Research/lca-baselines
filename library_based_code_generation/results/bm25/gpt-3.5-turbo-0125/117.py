```python
import blenderproc

# Initialize blenderproc library
blenderproc.init()

# Load 3D scene from .obj file and texture files
scene = blenderproc.load_mesh_objects_from_file(obj_file, texture_files)

# Label objects based on provided mapping
blenderproc.label_objects(scene, object_mapping)

# Separate walls, floors, and ceilings into distinct objects and assign labels
blenderproc.separate_and_label_surfaces(scene)

# Make lamp and ceiling objects emit light
blenderproc.make_objects_emit_light(scene, ['lamp', 'ceiling'])

# Create bounding volume hierarchy (BVH) tree containing all objects
bvh_tree = blenderproc.create_bvh_tree_multi_objects(scene)

# Sample camera locations and rotations above the floor
camera_pose = blenderproc.sample_camera_pose(scene)

# Enable normal, depth, and segmentation rendering
blenderproc.enable_rendering(scene, ['normal', 'depth', 'segmentation'])

# Render scene and write data to .hdf5 file in specified output directory
blenderproc.render_and_save(scene, output_directory)
```