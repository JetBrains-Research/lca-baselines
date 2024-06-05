```python
import numpy as np
import matplotlib.pyplot as plt
from pytransform3d.urdf import UrdfTransformManager
from pytransform3d.plot_utils import plot_transform
from pytransform3d.transformations import adjoint
from pytransform3d.rotations import matrix_from_screw_axis

def plot_wrench_transform(robot, joint_angles, wrench, screw_axis):
    tm = UrdfTransformManager()
    tm.load_urdf(robot)
    tm.set_joint_positions(joint_angles)
    
    plt.figure(figsize=(10, 10))
    
    plot_transform(ax=plt.gca(), A2B=np.eye(4), s=0.1)
    plot_transform(ax=plt.gca(), A2B=tm.get_transform("base_link", "tool0"), s=0.1)
    
    plot_transform(ax=plt.gca(), A2B=matrix_from_screw_axis(screw_axis), s=0.1)
    
    adj = adjoint(tm.get_transform("base_link", "tool0"))
    transformed_wrench = adj.T @ wrench
    
    plot_transform(ax=plt.gca(), A2B=matrix_from_screw_axis(transformed_wrench), s=0.1)
    
    plt.savefig("wrench_visualization.png")
    plt.show()

robot = "robot.urdf"
joint_angles = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
wrench = np.array([1.0, 2.0, 3.0, 0.1, 0.2, 0.3])
screw_axis = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])

plot_wrench_transform(robot, joint_angles, wrench, screw_axis)
```