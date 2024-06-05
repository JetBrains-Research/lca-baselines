 import pyaedt

pyaedt.set_non_graphical_mode()

app = pyaedt.Hfss()
app.new_project(name="Flex_CPWG", version="2022.1")
app.hfss.properties["material override"] = 1
app.hfss.properties["automatically use causal materials"] = 1
app.hfss.properties["open region"] = 1
app.hfss.properties["model units"] = "Millimeters"
app.hfss.mesh.initial_size = 5

total_length = 50  # total length of the flex cable CPWG
theta = 0.1  # bending angle
radius = 50  # bending radius
width = 0.5  # width of the signal line
height = 0.1  # height of the signal line
spacing = 0.1  # spacing between the signal line and the ground lines
ground_width = 1  # width of the ground lines
ground_thickness = 0.1  # thickness of the ground lines


def create_bending(radius, extension):
    bend = app.hfss.models["Model"].primitives.add_bend(
        radius, extension, app.hfss.models["Model"].primitives[-1].position
    )
    return bend


signal_line = app.hfss.models["Model"].primitives.add_polyline(
    [point_a(0, 0, 0), point_a(total_length, 0, 0)]
)
signal_line = test_54b_open_and_load_a_polyline(signal_line, app.hfss.models["Model"])

ground_lines = [
    app.hfss.models["Model"].primitives.add_polyline(
        [point_a(0, 0, 0), point_a(total_length, 0, 0)]
    ),
    app.hfss.models["Model"].primitives.add_polyline(
        [point_a(0, 0, 0), point_a(total_length, 0, 0)]
    ),
]
ground_lines = [
    test_54b_open_and_load_a_polyline(gl, app.hfss.models["Model"]) for gl in ground_lines
]

bend1 = create_bending(radius, theta * total_length)
bend2 = create_bending(radius, -theta * total_length)

signal_line = move_and_connect_to(signal_line, bend1, 1)
signal_line = move_and_connect_to(signal_line, bend2, 1)

for gl in ground_lines:
    gl = move_and_connect_to(gl, bend1, 1)
    gl = move_and_connect_to(gl, bend2, 1)

dielectric = app.icepak.models["Model"].primitives.add_box(
    0, 0, 0, total_length, height, spacing, "Dielectric"
)

bottom_metals = [
    app.hfss.models["Model"].primitives.add_box(
        0, 0, 0, total_length, 0.01, spacing, "Conductor"
    ),
    app.hfss.models["Model"].primitives.add_box(
        0, height + spacing, 0, total_length, 0.01, spacing, "Conductor"
    ),
]

app.hfss.models["Model"].primitives.unite(bottom_metals)

port1 = create_port_between_pin_and_layer(signal_line, bottom_metals[0], 1)
port2 = create_port_between_pin_and_layer(signal_line, bottom_metals[1], -1)

app.hfss.models["Model"].primitives.add_pebc(
    [
        app.hfss.models["Model"].primitives[-1].bounds[1],
        app.hfss.models["Model"].primitives[-1].bounds[3],
    ]
)

app.hfss.analysis.setup(
    exciter_type="Modal",
    frequency_domain_sweep_type="Linear",
    start_frequency=1,
    stop_frequency=10,
    number_of_points=101,
)

app.hfss.sweep.add_sweep_parameter("Frequency", "Frequency", "LinVar", 1, 10, 101)

app.hfss.analysis.solution_type = "Planar"

app.hfss.mesh.region.add_rectangular_region(
    0, 0, 0, total_length, height + spacing, spacing
)

app.hfss.mesh.region.set_edge_length(0.05)

app.hfss.mesh.region.set_adaptive_mesh_type(1)

app.hfss.mesh.region.set_adaptive_mesh_parameters(
    0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01
)

app.hfss.mesh.region.set_adaptive_mesh_iterations(5)

app.hfss.mesh.region.set_adaptive_mesh_error(0.01)

app.hfss.mesh.region.set_adaptive_mesh_threshold(0.01)

app.hfss.mesh.region.set_adaptive_mesh_growth_rate(1.2)

app.hfss.mesh.region.set_adaptive_mesh_min_edge_length(0.005)

app.hfss.mesh.region.set_adaptive_mesh_max_edge_length(0.1)

app.hfss.mesh.region.set_adaptive_mesh_max_passes(5)

app.hfss.mesh.region.set_adaptive_mesh_min_passes(2)

app.hfss.mesh.region.set_adaptive_mesh_convergence_threshold(0.01)

app.hfss.mesh.region.set_adaptive_mesh_convergence_iterations(5)

app.hfss.mesh.region.set_adaptive_mesh_convergence_type(1)

app.hfss.mesh.region.set_adaptive_mesh_convergence_error(0.01)

app.hfss.mesh.region.set_adaptive_mesh_convergence_growth_rate(1.2)

app.hfss.mesh.region.set_adaptive_mesh_convergence_min_edge_length(0.005)

app.hfss.mesh.region.set_adaptive_mesh_convergence_max_edge_length(0.1)

app.hfss.mesh.region.set_adaptive_mesh_convergence_max_passes(5)

app.hfss.mesh.region.set_adaptive_mesh_convergence_min_passes(2)

app.hfss.mesh.region.set_adaptive_mesh_convergence_threshold(0.01)

app.hfss.mesh.region.set_adaptive_mesh_convergence_iterations(5)

app.hfss.mesh.region.set_adaptive_mesh_convergence_type(1)

app.hfss.mesh.region.set_adaptive_mesh_convergence_error(0.01)

app.hfss.mesh.region.set_adaptive_mesh_convergence_growth_rate(1.2)

app.hfss.mesh.region.set_adaptive_mesh_convergence_min_edge_length(0.005)

app.hfss.mesh.region.set_adaptive_mesh_convergence_max_edge_length(0.1)

app.hfss.mesh.region.set_adaptive_mesh_convergence_max_pass