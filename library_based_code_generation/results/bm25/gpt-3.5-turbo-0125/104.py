```python
import numpy as np
import matplotlib.pyplot as plt
from pytransform3d.urdf import UrdfTransformManager
from pytransform3d.plot_utils import plot_transform
from pytransform3d.transformations import adjoint
from pytransform3d.plot_utils import plot_screw

def plot_wrench_transformation(robot, joint_angles, wrench):
    tm = UrdfTransformManager()
    tm.load_urdf(robot)
    tm.set_joint_positions(robot, joint_angles)
    
    tcp_frame = tm.get_frame(robot, "tcp")
    base_frame = tm.get_frame(robot, "base_link")
    
    plot_transform(tcp_frame, alpha=0.6)
    plot_transform(base_frame, alpha=0.6)
    
    plot_screw(wrench, frame=tcp_frame, alpha=0.6)
    
    adj = adjoint(tcp_frame, base_frame)
    transformed_wrench = adj.T @ wrench
    
    plot_screw(transformed_wrench, frame=base_frame, alpha=0.6)
    
    plt.savefig("wrench_transformation.png")
    plt.show()

robot = "robot.urdf"
joint_angles = [0, 0, 0, 0, 0, 0]
wrench = np.array([1, 2, 3, 0, 0, 0])

plot_wrench_transformation(robot, joint_angles, wrench)
```