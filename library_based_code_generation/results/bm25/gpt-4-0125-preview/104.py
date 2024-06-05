```python
import numpy as np
import matplotlib.pyplot as plt
from pytransform3d.urdf import UrdfTransformManager
from pytransform3d.transformations import transform_from, adjoint_from_transform
from pytransform3d.visualizer import plot_transform, plot_vector, plot_sphere

def plot_wrench(ax, position, force, color='r', label=''):
    """Plot a wrench as a force vector at a given position."""
    plot_vector(ax, position, force, color=color, label=label)

def transform_wrench(wrench, transform):
    """Transform a wrench from one frame to another using the adjoint representation."""
    adjoint = adjoint_from_transform(transform)
    return np.dot(adjoint, wrench)

def main(urdf_file, joint_angles):
    # Load robot model
    tm = UrdfTransformManager()
    with open(urdf_file, 'r') as file:
        tm.load_urdf(file.read())

    # Set robot joint angles
    for joint_name, angle in joint_angles.items():
        tm.set_joint(joint_name, angle)

    # Define wrench at TCP (force in N, torque in Nm)
    wrench_tcp = np.array([10, 0, 0, 0, 0, 1])  # Example wrench

    # Transformation from TCP to robot base
    transform_tcp_to_base = tm.get_transform('base_link', 'tcp')

    # Transform wrench to base frame
    wrench_base = transform_wrench(wrench_tcp, transform_tcp_to_base)

    # Visualization
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # Plot robot
    tm.plot_frames_in('base_link', s=0.1, ax=ax)
    tm.plot_connections_in('base_link', ax=ax)

    # Plot TCP frame
    plot_transform(ax, A2B=transform_tcp_to_base, s=0.1)

    # Plot wrench at TCP and base
    plot_wrench(ax, transform_tcp_to_base[:3, 3], wrench_tcp[:3], 'r', 'Wrench at TCP')
    plot_wrench(ax, np.zeros(3), wrench_base[:3], 'b', 'Wrench at Base')

    # Plot spherical mass at TCP
    plot_sphere(ax, radius=0.05, p=transform_tcp_to_base[:3, 3], color='g', alpha=0.5)

    # Save visualization
    plt.savefig('robot_wrench_visualization.png')
    plt.show()

if __name__ == "__main__":
    urdf_file = 'path_to_urdf_file.urdf'
    joint_angles = {'joint1': 0.5, 'joint2': -0.75, 'joint3': 1.2, 'joint4': -0.5, 'joint5': 0.3, 'joint6': -1.0}
    main(urdf_file, joint_angles)
```