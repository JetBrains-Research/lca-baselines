```python
import sys
from silx.gui import qt
from silx.gui.widgets import WaitingPushButton, ThreadPoolPushButton, RangeSlider, LegendIconWidget, ElidedLabel
from silx.gui.plot import Plot1D
from silx.gui.plot.LegendSelector import LegendSelector
import numpy as np

class MainWindow(qt.QWidget):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.initUI()

    def initUI(self):
        layout = qt.QVBoxLayout()

        # WaitingPushButton
        self.waitingButton = WaitingPushButton(text="Waiting Button")
        self.waitingButton.clicked.connect(self.toggleWaitingState)
        layout.addWidget(self.waitingButton)

        # ThreadPoolPushButton
        self.threadPoolButton = ThreadPoolPushButton(text="Compute Power")
        self.threadPoolButton.setCallable(self.computePower, 2, 8)  # Example: 2^8
        layout.addWidget(self.threadPoolButton)

        # RangeSlider
        self.rangeSlider = RangeSlider()
        self.rangeSlider.valueChanged.connect(self.sliderValueChanged)
        layout.addWidget(self.rangeSlider)

        # LegendIconWidget
        self.legendIconWidget = LegendIconWidget()
        plot = Plot1D()
        plot.addCurve(range(10), np.sin(range(10)), legend="sin", color="red")
        self.legendIconWidget.setPlotWidget(plot)
        layout.addWidget(self.legendIconWidget)

        # ElidedLabel
        self.elidedLabel = ElidedLabel(text="This is a very long text that should be elided at the end.")
        layout.addWidget(self.elidedLabel)

        self.setLayout(layout)
        self.setWindowTitle("Silx Widgets Example")
        self.resize(400, 300)

    def toggleWaitingState(self):
        self.waitingButton.setWaiting(not self.waitingButton.isWaiting())

    def computePower(self, base, exponent):
        return base ** exponent

    def sliderValueChanged(self, value):
        print(f"Slider value changed: {value}")

if __name__ == "__main__":
    app = qt.QApplication(sys.argv)
    qt.QUiLoader.registerCustomWidget(LegendIconWidget)
    mainWindow = MainWindow()
    mainWindow.show()
    sys.exit(app.exec_())
```