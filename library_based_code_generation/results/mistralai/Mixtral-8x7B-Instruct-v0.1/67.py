 ```python
import pyaedt

pyaedt.set_launch_design_async(False)

with pyaedt.PyAedt(specified_version="2022.1", nongui=True) as app:
    # Launch AEDT and 2D Extractor
    app.new_project("CPWG_Design", "AEDT 2D Extractor Project")
    app.start_application("2DExtractor")

    # Define variables
    L = 1000  # length of the CPWG in um
    W = 100  # width of the signal line in um
    S = 50  # gap between signal line and ground in um
    ER = 6.75  # relative permittivity of the dielectric
    H = 10  # height of the dielectric in um
    T = 2  # thickness of the conformal coating in um

    # Create primitives
    app.current_application.active_view.create_rectangle("box1", 0, 0, L, H, is_selected=True)
    app.current_application.active_view.create_rectangle("box2", 0, H - T, L, H, is_selected=True)

    # Create a signal
    app.current_application.active_view.create_rectangle("signal", S, H / 2 - W / 2, S + W, H / 2 + W / 2, is_selected=True)

    # Create a coplanar ground
    app.current_application.active_view.create_rectangle("ground1", 0, H / 2 - S / 2, S, H / 2 + S / 2, is_selected=True)
    app.current_application.active_view.create_rectangle("ground2", L, H / 2 - S / 2, L + S, H / 2 + S / 2, is_selected=True)

    # Create a reference ground plane
    app.current_application.active_view.create_rectangle("ref_ground", -50, -50, L + 50, H + 50, is_selected=True)

    # Create a dielectric
    app.current_application.active_view.create_box("dielectric", 0, 0, L, H, ER, is_selected=True)

    # Create a conformal coating
    app.current_application.active_view.create_box("conformal_coating", 0, H - T, L, H, 3.5, is_selected=True)

    # Assign a conductor to the signal
    app.current_application.active_view.set_properties("signal", {("Conductor", "Name") : "Copper"})

    # Create a reference ground
    app.current_application.active_view.set_properties("ref_ground", {("Conductor", "Name") : "Copper"})

    # Assign the Huray model on the signal
    app.current_application.active_view.set_properties("signal", {("SurfaceRoughness", "Model") : "Huray"})

    # Create the setup
    setup = app.current_application.new_setup("Setup1", "Setup")

    # Analyze it
    setup.analyze()

    # Plot solution data
    app.current_application.plot_solution("Setup1", "Electric Field", "E", "XYData", "Z")

    # Save the project and close AEDT
    app.save_project()
    app.stop_application("2DExtractor")
```