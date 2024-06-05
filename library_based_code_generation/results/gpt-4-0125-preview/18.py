```python
import sys
from silx.gui import qt
from silx.gui.widgets import WaitingPushButton, ThreadPoolPushButton, RangeSlider, LegendIconWidget, ElidedLabel
from silx.gui.plot.Colors import Colormap
from silx.gui.plot.items.legend import LineLegendIcon, SymbolLegendIcon, ColormapLegendIcon

class MainWindow(qt.QWidget):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.initUI()

    def initUI(self):
        layout = qt.QVBoxLayout(self)

        # WaitingPushButton
        self.waitingButton = WaitingPushButton(text="Waiting Button")
        self.waitingButton.clicked.connect(self.toggleWaitingState)
        layout.addWidget(self.waitingButton)

        # ThreadPoolPushButton
        self.threadPoolButton = ThreadPoolPushButton(text="Compute 2^10")
        self.threadPoolButton.setWorker(self.computePower, 2, 10)
        layout.addWidget(self.threadPoolButton)

        # RangeSlider
        self.rangeSlider = RangeSlider()
        self.rangeSlider.valueChanged.connect(self.onRangeSliderValueChanged)
        layout.addWidget(self.rangeSlider)

        # LegendIconWidget
        legendLayout = qt.QHBoxLayout()
        lineIcon = LegendIconWidget(icon=LineLegendIcon())
        symbolIcon = LegendIconWidget(icon=SymbolLegendIcon())
        colormapIcon = LegendIconWidget(icon=ColormapLegendIcon(colormap=Colormap()))
        legendLayout.addWidget(lineIcon)
        legendLayout.addWidget(symbolIcon)
        legendLayout.addWidget(colormapIcon)
        layout.addLayout(legendLayout)

        # ElidedLabel
        self.elidedLabel = ElidedLabel(text="This is a very long text that should be elided at the end.")
        layout.addWidget(self.elidedLabel)

        self.setLayout(layout)
        self.setWindowTitle("Silx Widgets Example")
        self.setGeometry(300, 300, 500, 200)

    def toggleWaitingState(self):
        if self.waitingButton.isWaiting():
            self.waitingButton.stopWaiting()
        else:
            self.waitingButton.startWaiting()

    def computePower(self, base, exponent):
        return base ** exponent

    def onRangeSliderValueChanged(self):
        print(f"RangeSlider Value Changed: {self.rangeSlider.value()}")

if __name__ == '__main__':
    app = qt.QApplication(sys.argv)
    qt.QUiLoader.registerCustomWidget(MainWindow)
    ex = MainWindow()
    ex.show()
    sys.exit(app.exec_())
```