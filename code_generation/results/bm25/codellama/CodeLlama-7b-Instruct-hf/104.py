  Here is some sample Python code that implements the described functionality using the pytransform3d library:
```
import pytransform3d as pt
import numpy as np
import matplotlib.pyplot as plt

# Load robot model from URDF file
robot = pt.Robot.from_urdf('robot.urdf')

# Set joint angles
robot.set_joint_angles(np.array([0, 0, 0, 0, 0, 0]))

# Get transformation from TCP to base frame
tcp_to_base = robot.get_transformation('tcp', 'base')

# Get transformation from TCP to tool center point (TCP)
tcp_to_tcp = robot.get_transformation('tcp', 'tcp')

# Get transformation from base frame to TCP
base_to_tcp = robot.get_transformation('base', 'tcp')

# Get transformation from base frame to tool center point (TCP)
base_to_tcp = robot.get_transformation('base', 'tcp')

# Get transformation from TCP to spherical mass
tcp_to_mass = robot.get_transformation('tcp', 'mass')

# Get transformation from base frame to spherical mass
base_to_mass = robot.get_transformation('base', 'mass')

# Get wrench at TCP
wrench_tcp = robot.get_wrench('tcp')

# Get wrench at base frame
wrench_base = robot.get_wrench('base')

# Plot robot model and transformations
fig, ax = plt.subplots(1, 2, figsize=(10, 5))
ax[0].plot(robot.get_pose('tcp'), 'o-')
ax[0].set_xlabel('X')
ax[0].set_ylabel('Y')
ax[0].set_zlabel('Z')
ax[0].set_title('TCP')
ax[1].plot(robot.get_pose('base'), 'o-')
ax[1].set_xlabel('X')
ax[1].set_ylabel('Y')
ax[1].set_zlabel('Z')
ax[1].set_title('Base')

# Plot wrench at TCP
ax[0].plot(wrench_tcp.force, 'r-')
ax[0].plot(wrench_tcp.torque, 'g-')
ax[0].set_xlabel('Force')
ax[0].set_ylabel('Torque')
ax[0].set_zlabel('')
ax[0].set_title('Wrench at TCP')

# Plot wrench at base frame
ax[1].plot(wrench_base.force, 'r-')
ax[1].plot(wrench_base.torque, 'g-')
ax[1].set_xlabel('Force')
ax[1].set_ylabel('Torque')
ax[1].set_zlabel('')
ax[1].set_title('Wrench at Base')

# Save visualization as image
fig.savefig('visualization.png')
```
This code loads a robot model from a URDF file, sets joint angles, and plots the robot model and the transformations. It also calculates the wrench at the TCP and the base frame and plots it as a screw. Finally, it saves the visualization as an image.

Note that this is just sample code and may need to be modified to work with your specific robot and setup. Additionally, the pytransform3d library is a powerful tool for working with transformations and visualizations, but it may require some familiarity with the underlying mathematics to use effectively.