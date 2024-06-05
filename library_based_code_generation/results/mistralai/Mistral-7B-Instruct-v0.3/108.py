 Here is a simplified example of how you might structure this experiment using the `nplab` library and `tkinter` for the GUI. Note that this is a basic example and does not include error handling, data saving, or other features you might need for a real experiment.

```python
import nplab
import time
import tkinter as tk

class Experiment:
    def __init__(self):
        self.shutter = nplab.devices.Shutter()
        self.spectrometer = nplab.devices.Spectrometer()

    def irradiate(self, irradiation_time, wait_time):
        self.shutter.open()
        time.sleep(irradiation_time)
        self.shutter.close()
        time.sleep(wait_time)
        self.spectrometer.capture_spectrum()

class GUI:
    def __init__(self, experiment):
        self.experiment = experiment
        self.irradiation_time = tk.IntVar()
        self.wait_time = tk.IntVar()

        self.window = tk.Tk()
        self.window.title("Experiment GUI")

        self.irradiation_label = tk.Label(self.window, text="Irradiation Time (s)")
        self.irradiation_entry = tk.Entry(self.window, textvariable=self.irradiation_time)
        self.wait_label = tk.Label(self.window, text="Wait Time (s)")
        self.wait_entry = tk.Entry(self.window, textvariable=self.wait_time)

        self.start_button = tk.Button(self.window, text="Start", command=self.start)

        self.irradiation_label.pack()
        self.irradiation_entry.pack()
        self.wait_label.pack()
        self.wait_entry.pack()
        self.start_button.pack()

    def start(self):
        irradiation_time = self.irradiation_time.get()
        wait_time = self.wait_time.get()
        self.experiment.irradiate(irradiation_time, wait_time)

if __name__ == "__main__":
    experiment = Experiment()
    gui = GUI(experiment)
    gui.window.mainloop()

    # Dummy spectrometer and shutter for testing
    dummy_spectrum = [0] * 100  # Replace with actual spectrum data
    experiment.spectrometer.spectrum = dummy_spectrum
    experiment.shutter.open()
    time.sleep(5)
    experiment.shutter.close()
```

This code creates an `Experiment` class with a `Spectrometer` and a `Shutter`, and a `GUI` class that allows the user to input irradiation and wait times. The `start` method in the `GUI` class triggers the experiment to run. The experiment and GUI are then tested with dummy spectrometer and shutter data.