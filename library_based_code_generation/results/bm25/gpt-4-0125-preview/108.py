```python
import nplab
from nplab.instrument import Instrument
from nplab.experiment import Experiment, ExperimentWithGui
from nplab.ui.ui_tools import UiLoader
from PyQt5 import QtWidgets, QtCore
import time
import numpy as np

class DummyShutter(Instrument):
    def __init__(self):
        super(DummyShutter, self).__init__()
        self.state = 'closed'

    def open(self):
        self.state = 'open'
        print("Shutter opened")

    def close(self):
        self.state = 'closed'
        print("Shutter closed")

class DummySpectrometer(Instrument):
    def __init__(self):
        super(DummySpectrometer, self).__init__()

    def take_spectrum(self):
        print("Taking spectrum")
        return np.random.rand(100)  # Simulate a spectrum

class ShutterSpectrometerExperiment(Experiment):
    def __init__(self, shutter, spectrometer, irradiation_time=1, wait_time=1):
        super(ShutterSpectrometerExperiment, self).__init__()
        self.shutter = shutter
        self.spectrometer = spectrometer
        self.irradiation_time = irradiation_time
        self.wait_time = wait_time

    def run(self):
        self.shutter.open()
        time.sleep(self.irradiation_time)
        self.shutter.close()
        time.sleep(self.wait_time)
        spectrum = self.spectrometer.take_spectrum()
        print("Spectrum taken:", spectrum)

class ShutterSpectrometerExperimentGui(QtWidgets.QWidget, ExperimentWithGui):
    def __init__(self, experiment):
        super(ShutterSpectrometerExperimentGui, self).__init__()
        self.experiment = experiment
        self.setup_ui()

    def setup_ui(self):
        self.layout = QtWidgets.QVBoxLayout()
        self.irradiation_time_spinbox = QtWidgets.QDoubleSpinBox()
        self.irradiation_time_spinbox.setRange(0, 60)
        self.irradiation_time_spinbox.setValue(self.experiment.irradiation_time)
        self.wait_time_spinbox = QtWidgets.QDoubleSpinBox()
        self.wait_time_spinbox.setRange(0, 60)
        self.wait_time_spinbox.setValue(self.experiment.wait_time)
        self.start_button = QtWidgets.QPushButton("Start Experiment")
        self.start_button.clicked.connect(self.start_experiment)

        self.layout.addWidget(QtWidgets.QLabel("Irradiation Time (s):"))
        self.layout.addWidget(self.irradiation_time_spinbox)
        self.layout.addWidget(QtWidgets.QLabel("Wait Time (s):"))
        self.layout.addWidget(self.wait_time_spinbox)
        self.layout.addWidget(self.start_button)
        self.setLayout(self.layout)

    def start_experiment(self):
        self.experiment.irradiation_time = self.irradiation_time_spinbox.value()
        self.experiment.wait_time = self.wait_time_spinbox.value()
        self.experiment.run()

if __name__ == "__main__":
    app = QtWidgets.QApplication([])
    shutter = DummyShutter()
    spectrometer = DummySpectrometer()
    experiment = ShutterSpectrometerExperiment(shutter, spectrometer)
    gui = ShutterSpectrometerExperimentGui(experiment)
    gui.show()
    app.exec_()
```