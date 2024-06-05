 import pyaedt
from pyaedt import Hfss3dLayout

with pyaedt.temp_directory():
    temp_folder = pyaedt.get_temp_folder()
    print("Temporary folder path: ", temp_folder)
    pyaedt.download_utils.check_and_download_file(
        "https://github.com/HELI-X/pyaedt_examples/raw/main/hfss/example.aedt", temp_folder
    )
    my_aedt = pyaedt.Aedt(specified_version="2022.1", nongui=True, new_desktop_session=True)
    my_aedt.new_desktop_session = True
    my_aedt.set_units("mm")
    my_aedt.activate_desktop()
    my_hfss = Hfss3dLayout(my_aedt)
    my_hfss.open_project(os.path.join(temp_folder, "example.aedt"))
    if my_hfss.is_file_existing_and_released("example.aedt"):
        my_hfss.delete_project()
        my_hfss.save_project(os.path.join(temp_folder, "example.aedt"))
    setups_obj = my_hfss.hfss.active_setup
    print("Boundaries: ", setups_obj.boundaries)
    my_hfss.hfss.hide_all_nets()
    my_hfss.hfss.make_net_visible("Net1")
    my_hfss.hfss.make_net_visible("Net2")
    my_hfss.hfss.plot_nets(["Net1", "Net2"])
    my_hfss.hfss.show_all_layers()
    my_hfss.hfss.set_layer_color("Layer1", "Red")
    my_hfss.hfss.hide_components_for_layers(["Top", "Bottom"])
    my_hfss.hfss.fit_all()
    my_hfss.hfss.close_project()
    my_aedt.release_desktop()