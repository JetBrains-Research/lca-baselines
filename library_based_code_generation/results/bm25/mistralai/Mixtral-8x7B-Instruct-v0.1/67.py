 ```python
import pyaedt
from pyaedt.generic.general_functions import CreateAedtObj, CreateAedtApplication
from pyaedt.generic.apps.aao_app import hfss3dl_a, icepak_a
from pyaedt.generic.apps.aeds_app import test_54b_open_and_load_a_polyline, test_09b_export_report_A
from pyaedt.generic.apps.epw_app import create_sheet_to_ground, _create_conformal
from pyaedt.generic.apps.epw_app import create_waveguide, create_plane
from pyaedt.generic.apps.epw_app import create, create_setup, _create_setup
from pyaedt.generic.apps.epw_app import source_terms_to_ground, CreateAedtBlockObj
from pyaedt.generic.apps.epw_app import signal, create_dataset1d_project
from pyaedt.generic.apps.epw_app import create_new_project, create_3d_plot

pyaedt.set_log_level("ERROR")
app = CreateAedtApplication("2D Extractor")
app.start()
app_name = "CPWG_Project"
new_project = create_new_project(app, app_name, "2D Extractor")
aedt_obj = CreateAedtObj(new_project, app)
aedt_obj.create_project()

# Define variables
width = 0.5 