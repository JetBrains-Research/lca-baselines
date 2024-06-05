```python
import numpy as np
from pytransform3d.urdf import UrdfTransformManager
from pytransform3d.transformations import transform_from, plot_transform
from pytransform3d.visualizer import UrdfVisualizer
from pytransform3d.rotations import matrix_from_axis_angle
from pytransform3d.transform_manager import TransformManager

def plot_wrench(ax, position, force, color='r', scale=0.1):
    ax.quiver(position[0], position[1], position[2], force[0], force[1], force[2], color=color, length=scale, normalize=True)

def transform_wrench(wrench, transformation):
    adjoint = np.zeros((6, 6))
    R = transformation[:3, :3]
    p = transformation[:3, 3]
    adjoint[:3, :3] = R
    adjoint[3:, 3:] = R
    adjoint[3:, :3] = np.dot(skew(p), R)
    return np.dot(adjoint, wrench)

def skew(vector):
    return np.array([[0, -vector[2], vector[1]],
                     [vector[2], 0, -vector[0]],
                     [-vector[1], vector[0], 0]])

def main():
    # Load robot model
    tm = UrdfTransformManager()
    with open("path_to_urdf_file.urdf", "r") as f:
        urdf_string = f.read()
    tm.load_urdf(urdf_string)
    
    # Set joint angles
    joint_angles = {"joint1": 0.5, "joint2": 1.2, "joint3": -0.5, "joint4": 0.4, "joint5": 1.1, "joint6": -0.3}
    for joint, angle in joint_angles.items():
        tm.set_joint(joint, angle)
    
    # Visualize robot
    visualizer = UrdfVisualizer("path_to_urdf_file.urdf", tm=tm, mesh_path=".")
    fig = visualizer.create_figure(size=(10, 10))
    ax = visualizer.create_3d_axis(fig)
    visualizer.plot_robot(ax, joint_angles)
    
    # Define wrench at TCP (force, torque)
    wrench_tcp = np.array([1, 0, 0, 0, 0, 1])  # Example wrench
    
    # Transformation from TCP to base
    tcp_to_base = tm.get_transform("base_link", "tcp_link")
    
    # Transform wrench to base frame
    wrench_base = transform_wrench(wrench_tcp, tcp_to_base)
    
    # Plot wrench at TCP
    tcp_position = tcp_to_base[:3, 3]
    plot_wrench(ax, tcp_position, wrench_tcp[:3], color='r')
    
    # Plot wrench at base
    base_position = np.array([0, 0, 0])
    plot_wrench(ax, base_position, wrench_base[:3], color='b')
    
    # Save visualization
    fig.savefig("robot_wrench_visualization.png")

if __name__ == "__main__":
    main()
```