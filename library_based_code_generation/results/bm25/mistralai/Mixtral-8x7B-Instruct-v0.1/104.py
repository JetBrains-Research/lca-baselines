 ```python
import pytransform3d.transformations as tf
import pytransform3d.visualizer as pv
import pytransform3d.robot as pr
import pytransform3d.screws as ps
import trimesh
import os
import numpy as np

def plot_transformation(T_AB, a, w, frame_size=0.1):
    screw_axis_A = ps.screw_axis_from_parameters(a, w)
    screw_A = ps.Screw(screw_axis_A)
    arrow_A = pv.Arrow(screw_A.n, screw_A.d, shaft_radius=0.005, head_radius=0.015, head_length=0.05, color='r')
    arrow_A.transform = T_AB
    pv.plot_transformations([arrow_A], frame_size=frame_size)

def wrench_visualization(urdf_path, joint_angles, force, torque, save_image=False):
    robot = pr.Robot.from_urdf_file(urdf_path)
    robot.set_joint_angles(joint_angles)
    T_BT = robot.get_transform()
    T_TCPB = T_BT * robot.get_end_effector_transform()
    T_WTCP = tf.compose_transforms(*[T_TCPB, ps.screw_to_transform(ps.Screw(ps.screw_axis_from_parameters(np.zeros(6), [0, 0, 0, force, torque])))])
    T_WB = tf.compose_transforms(*[T_BT, ps.adjoint_transform(T_TCPB, ps.Screw(ps.screw_axis_from_parameters(np.zeros(6), [0, 0, 0, force, torque])))])
    screw_WTCP = ps.Screw(ps.screw_axis_from_transform(T_WTCP))
    screw_WB = ps.Screw(ps.screw_axis_from_transform(T_WB))
    arrow_TCP = pv.Arrow(np.zeros(3), 0.05, shaft_radius=0.01, head_radius=0.03, head_length=0.05, color='g')
    arrow_TCP.transform = T_TCPB
    arrow_WTCP = pv.Arrow(screw_WTCP.n, screw_WTCP.d, shaft_radius=0.005, head_radius=0.015, head_length=0.05, color='r')
    arrow_WTCP.transform = T_WTCP
    arrow_WB = pv.Arrow(screw_WB.n, screw_WB.d, shaft_radius=0.005, head_radius=0.015, head_length=0.05, color='b')
    arrow_WB.transform = T_WB
    mesh_sphere = trimesh.creation.icosphere(radius=0.02)
    mesh_sphere.apply_transform(T_WTCP)
    pv.plot_transformations([robot.visual_objects, arrow_TCP, arrow_WTCP, arrow_WB, mesh_sphere], frame_size=0.1)
    if save_image:
        pv.save_image('wrench_visualization.png')

urdf_path = 'path/to/urdf/file'
joint_angles = [0, 0, 0, 0, 0, 0]
force = 10
torque = 5
wrench_visualization(urdf_path, joint_angles, force, torque, save_image=True)
```