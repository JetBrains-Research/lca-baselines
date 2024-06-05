```python
import sys
from silx.gui import qt
from silx.gui.widgets.WaitingPushButton import WaitingPushButton
from silx.gui.widgets.ThreadPoolPushButton import ThreadPoolPushButton
from silx.gui.widgets.RangeSlider import RangeSlider
from silx.gui.plot.LegendIconWidget import LegendIconWidget
from silx.gui.widgets.ElidedLabel import ElidedLabel
from silx.utils import exception

class MyApplication(qt.QMainWindow):
    def __init__(self):
        super(MyApplication, self).__init__()

        self.setWindowTitle('Simple GUI Application')
        self.setGeometry(100, 100, 800, 600)

        waitingPushButton = WaitingPushButton('Waiting Push Button')
        waitingPushButton.clicked.connect(self.toggleWaitingState)

        threadPoolPushButton = ThreadPoolPushButton('Thread Pool Push Button')
        threadPoolPushButton.clicked.connect(self.computePowerOperation)

        rangeSlider = RangeSlider('Range Slider')
        rangeSlider.sigValueChanged.connect(self.handleRangeSliderValueChanged)
        rangeSlider.sigPositionChanged.connect(self.handleRangeSliderPositionChanged)

        legendIconWidget = LegendIconWidget('Legend Icon Widget')

        elidedLabel = ElidedLabel('Elided Label')
        elidedLabel.setText('This is a long text that will be elided if it exceeds the available space.')

        layout = qt.QVBoxLayout()
        layout.addWidget(waitingPushButton)
        layout.addWidget(threadPoolPushButton)
        layout.addWidget(rangeSlider)
        layout.addWidget(legendIconWidget)
        layout.addWidget(elidedLabel)

        centralWidget = qt.QWidget()
        centralWidget.setLayout(layout)
        self.setCentralWidget(centralWidget)

    def toggleWaitingState(self):
        sender = self.sender()
        sender.setWaiting(not sender.isWaiting())

    def computePowerOperation(self):
        # Perform power operation here
        pass

    def handleRangeSliderValueChanged(self, value):
        print('Range Slider value changed:', value)

    def handleRangeSliderPositionChanged(self, position):
        print('Range Slider position changed:', position)

if __name__ == '__main__':
    app = qt.QApplication(sys.argv)
    with exception.handler():
        window = MyApplication()
        window.show()
        sys.exit(app.exec_())
```