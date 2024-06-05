import nplab
from nplab.instrument.shutter import Shutter
from nplab.instrument.spectrometer import Spectrometer
from nplab.ui import ExperimentUI

shutter = Shutter(dummy=True)
spectrometer = Spectrometer(dummy=True)

def experiment(iri_time, wait_time):
    shutter.open()
    nplab.sleep(wait_time)
    spectrum = spectrometer.take_spectrum()
    return spectrum

ui = ExperimentUI(experiment)
ui.add_parameter('Irradiation Time', 'iri_time', min=0, max=10, step=0.1)
ui.add_parameter('Wait Time', 'wait_time', min=0, max=10, step=0.1)
ui.add_data_browser()
ui.add_spectrometer_controls(spectrometer)
ui.add_shutter_controls(shutter)

ui.run()