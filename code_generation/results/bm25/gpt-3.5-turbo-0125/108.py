```python
import nplab
from nplab.instruments import Shutter, Spectrometer
from nplab.experiment import ExperimentWithGui

class MyExperiment(ExperimentWithGui):
    def __init__(self):
        super().__init__()
        self.shutter = Shutter()
        self.spectrometer = Spectrometer()

    def run(self):
        self.shutter.open()
        self.wait(self.irradiation_time)
        spectrum = self.spectrometer.take_spectrum()
        self.shutter.close()
        return spectrum

    def setup_gui(self):
        self.add_parameter('irradiation_time', label='Irradiation Time', unit='s', default=1)
        self.add_parameter('wait_time', label='Wait Time', unit='s', default=1)
        self.add_data_browser()
        self.add_spectrometer_controls(self.spectrometer)
        self.add_shutter_controls(self.shutter)

experiment = MyExperiment()
experiment.setup_gui()
experiment.run()
```