



 Here's an example of Python code that meets the requirements:

```
import numpy as np
import pytransform3d.transformations as pt
import pytransform3d.rotations as pr
import pytransform3d.visualizer as pv
import pytransform3d.urdf as pyu

def plot_wrench(ax, wrench, tcp_frame, base_frame):
    """Plot the wrench in the TCP and base frames."""
    # Transform wrench from TCP to base frame
    wrench_base = pt.transform_from(
        pr.matrix_from_compact_axis_angle(tcp_frame[:3, :3]), tcp_frame[:3, 3]
    ).dot(wrench)

    # Plot wrench in TCP frame
    ax.plot_screw(
        tcp_frame[:3, 3],
        tcp_frame[:3, :3].dot(wrench[:3]),
        tcp_frame[:3, :3].dot(wrench[3:]),
        color="red",
    )

    # Plot wrench in base frame
    ax.plot_screw(
        base_frame[:3, 3],
        base_frame[:3, :3].dot(wrench_base[:3]),
        base_frame[:3, :3].dot(wrench_base[3:]),
        color="blue",
    )

def main():
    # Load robot model from URDF file
    robot = pyu.URDF.from_xml_file("path/to/robot.urdf")

    # Set joint angles
    joint_angles = [0.5, -0.3, 0.2, 0.3, -0.2, 0.1]
    robot.set_joint_positions(joint_angles, robot.joint_names)

    # Create figure
    fig = pv.figure()
    ax = fig.plot_basis(s=0.1)

    # Plot robot model
    robot.plot(ax, s=0.1)

    # Define TCP frame
    tcp_frame = pt.transform_from(
        pr.active_matrix_from_intrinsic_euler_xyz(np.array([0.5, -0.3, 0.2])),
        np.array([0.2, 0.3, 0.4]),
    )

    # Define base frame
    base_frame = robot.link_transforms[robot.base_link]

    # Define wrench
    wrench = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 1.0])

    # Plot wrench
    plot_wrench(ax, wrench, tcp_frame, base_frame)

    # Plot TCP
    ax.plot_transform(tcp_frame, s=0.1)

    # Plot spherical mass
    ax.plot_sphere(tcp_frame[:3, 3], radius=0.05, c="black")

    # Save figure as image
    fig.save_image("figure.png")

if __name__ == "__main__":
    main()
```

This code defines a function `plot_wrench` that plots the wrench in the TCP and base frames. It then loads a robot model from a URDF file, sets joint angles, creates a figure, plots the robot model, defines the TCP frame, the base frame, and the wrench, and plots the wrench in both frames. Finally, it plots the TCP, plots a spherical mass, and saves the figure as an image.