import os
from pyaedt import Maxwell2d

project = Maxwell2d()
project._non_graphical = True
design = project.insert_design("Maxwell 2D")
project.save_project()

rect1 = design.modeler.primitives.create_rectangle([0, 0, 0], [10, 5], name="Rectangle1")
rect2 = design.modeler.primitives.duplicate(rect1)
air_region = design.modeler.create_air_region()
design.assign_windings([rect1, rect2])
design.assign_balloon(air_region, [rect1, rect2])
design.modeler.plot_model()

setup = design.create_setup("Trans1")
setup.create_rectangular_plot("MyPlot")
design.analyze_nominal()
design.post.create_output()
design.post.plot_fields()

project.plot_fields()
project.clean_cache_and_files()
project.close_project()