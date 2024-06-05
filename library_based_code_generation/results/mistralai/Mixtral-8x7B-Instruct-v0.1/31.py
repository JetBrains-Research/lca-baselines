 ```python
import pyaedt

pyaedt.set_launch_application("HFSS")
pyaedt.start_application()

HFSS = pyaedt.HFSS()
HFSS.new_project("Flex_CPWG", "HFSS")
HFSS.set_project_property("nonGraphical", "true")
HFSS.set_project_property("materialOverride", "true")
HFSS.set_project_property("causalMaterials", "true")
HFSS.set_project_property("openRegion", "true")
HFSS.set_project_property("modelUnits", "Millimeters")
HFSS.set_project_property("initialMesh", "true")
HFSS.set_project_property("analysisType", "Planar")
HFSS.set_project_property("frequency_step", "0.1GHz,0.1GHz,30")

total_length = 50  # total length of the CPWG
theta = 0.1  # bending angle
radius = 10  # bending radius
width = 1  # width of the signal line
height = 0.5  # height of the CPWG
spacing = 1  # spacing between the signal line and the ground lines
ground_width = 2  # width of the ground lines
ground_thickness = 0.5  # thickness of the ground lines


def create_bending(radius, extension):
    return HFSS.models["Model"].curves.add_circle(
        (0, 0, 0), radius, extension, 128, "degrees"
    )


signal_line = HFSS.models["Model"].primitives.add_polyline(
    [
        (0, 0, 0),
        (total_length / 2, 0, 0),
        (total_length / 2, radius * (1 - math.cos(math.radians(theta))), radius * math.sin(math.radians(theta))),
        (total_length / 2 - radius * (1 - math.cos(math.radians(theta))), radius, 0),
        (0, radius, 0),
    ]
)
ground_lines = HFSS.models["Model"].primitives.add_polyline(
    [
        (0, 0, 0),
        (total_length / 2, 0, 0),
        (total_length / 2, radius * (1 - math.cos(math.radians(theta))), radius * math.sin(math.radians(theta))),
        (total_length / 2 - radius * (1 - math.cos(math.radians(theta))), radius, 0),
        (0, radius, 0),
        (0, 0, 0),
    ]
)

dielectric = HFSS.models["Model"].primitives.add_box(
    (0, 0, 0), total_length, radius, height, HFSS.get_material("FR4")
)
bottom_metals = HFSS.models["Model"].primitives.add_box(
    (0, -ground_thickness, 0), total_length, radius + ground_width, height + ground_thickness, HFSS.get_material("Copper")
)

HFSS.models["Model"].set_material_override("Copper", "PEC")
HFSS.models["Model"].set_material_override("FR4", "FR4")

port_interfaces = HFSS.models["Model"].primitives.add_peccircle(
    [(0, 0, 0), (0, radius, 0)], 0.5, "PEC"
)
perfect_e = HFSS.models["Model"].primitives.add_peccircle(
    [(total_length, radius, 0), (total_length, 0, 0)], 0.5, "PEC"
)

ports = HFSS.models["Model"].ports.add_port_list(
    [
        HFSS.models["Model"].ports.add_port(port_interfaces[0], 1),
        HFSS.models["Model"].ports.add_port(perfect_e[0], 2),
    ]
)

HFSS.models["Model"].analyze()
HFSS.cleanup()
```
Please note that this code is a starting point and may need to be adjusted based on the specific version of AEDT and the pyaedt library being used. Additionally, the code assumes that the pyaedt library has been properly installed and configured.