```python
import sys
import numpy as np
import pyqtgraph as pg
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QGridLayout
from PyQt5.QtCore import QTimer

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Basic plotting examples")
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        layout = QGridLayout()
        self.central_widget.setLayout(layout)

        # Basic array plotting
        self.plotWidget1 = pg.PlotWidget()
        self.plotWidget1.plot(np.random.normal(size=100))
        layout.addWidget(self.plotWidget1, 0, 0)

        # Multiple curves
        self.plotWidget2 = pg.PlotWidget()
        x = np.arange(100)
        self.plotWidget2.plot(x, np.sin(x))
        self.plotWidget2.plot(x, np.cos(x))
        layout.addWidget(self.plotWidget2, 0, 1)

        # Drawing with points
        self.plotWidget3 = pg.PlotWidget()
        self.plotWidget3.plot(np.random.normal(size=100), pen=None, symbol='o')
        layout.addWidget(self.plotWidget3, 1, 0)

        # Parametric plot with grid enabled
        self.plotWidget4 = pg.PlotWidget()
        t = np.linspace(0, 2*np.pi, 1000)
        self.plotWidget4.plot(np.sin(t), np.cos(t))
        self.plotWidget4.showGrid(x=True, y=True)
        layout.addWidget(self.plotWidget4, 1, 1)

        # Scatter plot with axis labels and log scale
        self.plotWidget5 = pg.PlotWidget()
        self.plotWidget5.setLogMode(x=True, y=True)
        self.plotWidget5.setLabel('left', 'Y Axis', units='A')
        self.plotWidget5.setLabel('bottom', 'X Axis', units='s')
        self.plotWidget5.plot(np.random.exponential(size=100), np.random.exponential(size=100), pen=None, symbol='x')
        layout.addWidget(self.plotWidget5, 2, 0)

        # Updating plot
        self.plotWidget6 = pg.PlotWidget()
        self.data6 = np.random.normal(size=100)
        self.curve6 = self.plotWidget6.plot(self.data6)
        layout.addWidget(self.plotWidget6, 2, 1)

        # Filled plot with axis disabled
        self.plotWidget7 = pg.PlotWidget()
        self.plotWidget7.plot(np.random.normal(size=100), fillLevel=0, brush=(0,0,255,150))
        self.plotWidget7.hideAxis('bottom')
        self.plotWidget7.hideAxis('left')
        layout.addWidget(self.plotWidget7, 3, 0)

        # Region selection and zoom on selected region
        self.plotWidget8 = pg.PlotWidget()
        self.region = pg.LinearRegionItem()
        self.region.setZValue(10)
        self.plotWidget8.addItem(self.region)
        self.region.sigRegionChanged.connect(self.updateRegion)
        self.plotWidget8.plot(np.random.normal(size=100))
        layout.addWidget(self.plotWidget8, 3, 1)

        # Timer for updating plot
        self.timer = QTimer()
        self.timer.timeout.connect(self.updatePlot)
        self.timer.start(1000)

    def updatePlot(self):
        self.data6 = np.roll(self.data6, -1)
        self.data6[-1] = np.random.normal()
        self.curve6.setData(self.data6)

    def updateRegion(self):
        minX, maxX = self.region.getRegion()
        self.plotWidget8.setXRange(minX, maxX, padding=0)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    main = MainWindow()
    main.show()
    sys.exit(app.exec_())
```