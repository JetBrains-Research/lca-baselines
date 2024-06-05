import os
from pyaedt import Maxwell2d

maxwell = Maxwell2d()
maxwell._non_graphical = True

maxwell.insert_design("Maxwell2DDesign1")
maxwell.save_project(os.path.join(maxwell.working_directory, "Maxwell2DProject.aedt"))

maxwell.modeler.primitives.create_rectangle([0, 0, 0], [10, 5], name="Rectangle1")
maxwell.modeler.primitives.duplicate("Rectangle1", [15, 0])
maxwell.modeler.assign_air_region()
maxwell.modeler.assign_windings(["Rectangle1", "Rectangle1_1"])
maxwell.modeler.assign_balloon("Air")

maxwell.plot_model()

maxwell.create_setup("Trans1", time_per_step=1e-6, num_steps=1000)
maxwell.post.create_rectangular_plot("Trans1", "Surface Loss Density")
maxwell.analyze_nominal()

maxwell.post.plot_fields_on_cuts("Surface Loss Density")

maxwell.close_desktop()