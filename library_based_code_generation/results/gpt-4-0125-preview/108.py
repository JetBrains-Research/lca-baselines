```python
import nplab
import nplab.instrument.spectrometer as sp
import nplab.instrument.shutter as sh
from nplab.ui.ui_tools import DataBrowser, ExperimentUI
import time
from PyQt5 import QtWidgets, QtCore

class SpectrometerExperiment(nplab.Experiment):
    def __init__(self, spectrometer, shutter, irradiation_time=1, wait_time=1):
        super().__init__()
        self.spectrometer = spectrometer
        self.shutter = shutter
        self.irradiation_time = irradiation_time
        self.wait_time = wait_time

    def run(self):
        self.shutter.open()
        time.sleep(self.irradiation_time)
        self.shutter.close()
        time.sleep(self.wait_time)
        spectrum = self.spectrometer.take_spectrum()
        self.datafile.create_dataset("spectrum", data=spectrum)

class SpectrometerExperimentUI(QtWidgets.QWidget):
    def __init__(self, experiment):
        super().__init__()
        self.experiment = experiment
        self.setup_ui()

    def setup_ui(self):
        self.layout = QtWidgets.QVBoxLayout()
        self.irradiation_time_spinbox = QtWidgets.QDoubleSpinBox()
        self.wait_time_spinbox = QtWidgets.QDoubleSpinBox()
        self.start_button = QtWidgets.QPushButton("Start Experiment")
        self.irradiation_time_spinbox.setRange(0, 60)
        self.wait_time_spinbox.setRange(0, 60)
        self.irradiation_time_spinbox.setValue(self.experiment.irradiation_time)
        self.wait_time_spinbox.setValue(self.experiment.wait_time)

        self.layout.addWidget(QtWidgets.QLabel("Irradiation Time (s):"))
        self.layout.addWidget(self.irradiation_time_spinbox)
        self.layout.addWidget(QtWidgets.QLabel("Wait Time (s):"))
        self.layout.addWidget(self.wait_time_spinbox)
        self.layout.addWidget(self.start_button)

        self.start_button.clicked.connect(self.start_experiment)
        self.setLayout(self.layout)

    def start_experiment(self):
        self.experiment.irradiation_time = self.irradiation_time_spinbox.value()
        self.experiment.wait_time = self.wait_time_spinbox.value()
        self.experiment.run()

class DummySpectrometer:
    def take_spectrum(self):
        return [1, 2, 3, 4, 5]  # Dummy spectrum data

class DummyShutter:
    def open(self):
        print("Shutter opened")

    def close(self):
        print("Shutter closed")

if __name__ == "__main__":
    app = QtWidgets.QApplication([])
    spectrometer = DummySpectrometer()
    shutter = DummyShutter()
    experiment = SpectrometerExperiment(spectrometer, shutter)
    ui = SpectrometerExperimentUI(experiment)
    ui.show()
    app.exec_()
```