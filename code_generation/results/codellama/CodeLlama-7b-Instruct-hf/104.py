  import pytransform3d.all as pt
import pytransform3d.urdf as ptu
import matplotlib.pyplot as plt

# Load robot model from URDF file
robot = ptu.RobotFromURDF("robot.urdf")

# Set joint angles
robot.q = [0, 0, 0, 0, 0, 0]

# Get transformation from TCP to base frame
T_tcp_to_base = robot.fkine(robot.q)

# Get transformation from TCP to screw axis
T_tcp_to_screw = pt.translation_matrix([0, 0, 0]) @ pt.rotation_matrix_from_euler([0, 0, 0])

# Get transformation from screw axis to base frame
T_screw_to_base = T_tcp_to_base @ T_tcp_to_screw

# Get wrench in TCP frame
wrench_tcp = pt.wrench_from_force_and_torque(force=[1, 0, 0], torque=[0, 0, 1])

# Transform wrench from TCP frame to base frame
wrench_base = T_screw_to_base @ wrench_tcp

# Plot robot model and transformations
fig, ax = plt.subplots(1, 2, figsize=(10, 5))
ax[0].add_artist(robot.plot(ax=ax[0], color="gray"))
ax[0].set_title("Robot Model")
ax[1].add_artist(pt.plot_screw(wrench_tcp, ax=ax[1], color="red"))
ax[1].set_title("Wrench in TCP Frame")
ax[1].set_xlabel("Force")
ax[1].set_ylabel("Torque")
ax[1].set_zlabel("Magnitude")
ax[1].set_xlim([-1, 1])
ax[1].set_ylim([-1, 1])
ax[1].set_zlim([-1, 1])
ax[1].view_init(elev=30, azim=45)

ax[0].add_artist(pt.plot_screw(wrench_base, ax=ax[0], color="blue"))
ax[0].set_title("Wrench in Base Frame")
ax[0].set_xlabel("Force")
ax[0].set_ylabel("Torque")
ax[0].set_zlabel("Magnitude")
ax[0].set_xlim([-1, 1])
ax[0].set_ylim([-1, 1])
ax[0].set_zlim([-1, 1])
ax[0].view_init(elev=30, azim=45)

# Save visualization as image
fig.savefig("visualization.png")