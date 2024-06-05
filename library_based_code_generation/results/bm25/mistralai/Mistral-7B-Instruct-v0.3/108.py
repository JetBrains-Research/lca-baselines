 Here is a basic example of how you might structure this experiment using the nplab library. Please note that this is a simplified version and you may need to adjust it according to your specific needs. Also, I'm assuming that you have the necessary classes and functions for the GUI and the dummy spectrometer and shutter.

```python
from nplab.experiment import Experiment, ExperimentGuiMixin
from nplab.instrument import InstrumentA
from nplab.gui import ExperimentGui, ExperimentWithGui, ScanningExperiment
from nplab.controls import PinAndClearButtons

class ShutterSpectrometerExperiment(Experiment, ExperimentGuiMixin):
    def __init__(self, spectrometer, shutter, **kwargs):
        super().__init__(**kwargs)
        self.spectrometer = spectrometer
        self.shutter = shutter

    def setup(self):
        self.add_instrument(self.spectrometer)
        self.add_instrument(self.shutter)
        self.add_control(PinAndClearButtons('Irradiation Time'))
        self.add_control(PinAndClearButtons('Wait Time'))

    def run(self):
        irradiation_time = self.get_control('Irradiation Time').value
        wait_time = self.get_control('Wait Time').value
        self.shutter.open()
        self.wait(irradiation_time)
        self.spectrometer.take_spectrum()
        self.shutter.close()
        self.wait(wait_time)

class ShutterSpectrometerExperimentGui(ExperimentWithGui, ShutterSpectrometerExperiment):
    def __init__(self, spectrometer, shutter, **kwargs):
        super().__init__(ShutterSpectrometerExperiment(spectrometer, shutter, **kwargs), **kwargs)

    def build_gui(self):
        gui = ExperimentGui(self)
        gui.add_widget(self.spectrometer.get_gui())
        gui.add_widget(self.shutter.get_gui())
        gui.add_widget(self.get_control('Irradiation Time').get_gui())
        gui.add_widget(self.get_control('Wait Time').get_gui())
        return gui

# Dummy instruments
class DummySpectrometer(InstrumentA):
    def take_spectrum(self):
        print("Taking dummy spectrum")

class DummyShutter(InstrumentA):
    def open(self):
        print("Opening dummy shutter")

    def close(self):
        print("Closing dummy shutter")

# Test the experiment
spectrometer = DummySpectrometer()
shutter = DummyShutter()
experiment = ShutterSpectrometerExperiment(spectrometer, shutter, name='ShutterSpectrometerExperiment')
experiment_gui = ShutterSpectrometerExperimentGui(spectrometer, shutter, name='ShutterSpectrometerExperimentGui')
experiment_gui.run_experiment()
```

This code creates a new experiment called `ShutterSpectrometerExperiment` that uses a `DummySpectrometer` and a `DummyShutter`. The experiment opens the shutter, waits for a specified amount of time, takes a spectrum, closes the shutter, and waits for another specified amount of time. The experiment also has a GUI that includes the spectrometer controls, shutter controls, and controls for the irradiation time and wait time.

The `ShutterSpectrometerExperimentGui` class is a subclass of `ShutterSpectrometerExperiment` and `ExperimentWithGui`, which allows it to have a GUI. The GUI is built in the `build_gui` method and includes the spectrometer's GUI, the shutter's GUI, and the controls for the irradiation time and wait time.

Finally, the experiment and its GUI are tested using the dummy spectrometer and shutter.