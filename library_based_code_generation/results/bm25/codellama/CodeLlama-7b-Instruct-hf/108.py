  import nplab

# Define the experiment
experiment = nplab.Experiment(
    "Shutter and Spectrometer Experiment",
    [
        nplab.InstrumentA("Shutter", nplab.a_changed),
        nplab.InstrumentA("Spectrometer", nplab.a_changed)
    ]
)

# Define the user interface
experiment_gui = nplab.ExperimentGuiMixin(experiment)

# Define the experiment flow
experiment_flow = nplab.ScanningExperiment(
    experiment,
    [
        nplab.move_a_lot,
        nplab.flush_camera_and_wait,
        nplab.calculate_a_b_coefficients,
        nplab.plotHistAndFit,
        nplab.loss_centres_and_widths,
        nplab.optimize_peaks_and_bg
    ]
)

# Define the shutter controls
shutter_controls = nplab.PinAndClearButtons(
    experiment.instruments["Shutter"],
    "Shutter Controls",
    [
        nplab.InstrumentA("Open", nplab.a_changed),
        nplab.InstrumentA("Close", nplab.a_changed)
    ]
)

# Define the spectrometer controls
spectrometer_controls = nplab.PinAndClearButtons(
    experiment.instruments["Spectrometer"],
    "Spectrometer Controls",
    [
        nplab.InstrumentA("Irradiate", nplab.a_changed),
        nplab.InstrumentA("Wait", nplab.a_changed)
    ]
)

# Define the data browser
data_browser = nplab.DataBrowser(
    experiment,
    "Data Browser",
    [
        nplab.InstrumentA("Spectrum", nplab.a_changed)
    ]
)

# Define the experiment and its GUI
experiment_with_gui = nplab.ExperimentWithGui(
    experiment,
    experiment_gui,
    experiment_flow,
    shutter_controls,
    spectrometer_controls,
    data_browser
)

# Test the experiment and its GUI using dummy instruments
experiment_with_gui.test(
    nplab.DummyInstrument("Shutter"),
    nplab.DummyInstrument("Spectrometer")
)

# Run the experiment and its GUI
experiment_with_gui.run()