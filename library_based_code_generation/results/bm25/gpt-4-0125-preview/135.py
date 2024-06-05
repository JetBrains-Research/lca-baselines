```python
import sys
import numpy as np
import pyqtgraph as pg
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QGridLayout
from PyQt5.QtCore import QTimer

class PlottingWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Basic plotting examples")
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QGridLayout(self.central_widget)

        self.initUI()

    def initUI(self):
        # Basic array plotting
        self.plotWidget1 = pg.PlotWidget(title="Basic Array Plotting")
        self.plotWidget1.plot(np.random.normal(size=100))
        self.layout.addWidget(self.plotWidget1, 0, 0)

        # Multiple curves
        self.plotWidget2 = pg.PlotWidget(title="Multiple Curves")
        for i in range(3):
            self.plotWidget2.plot(np.random.normal(size=100) + i * 2)
        self.layout.addWidget(self.plotWidget2, 0, 1)

        # Drawing with points
        self.plotWidget3 = pg.PlotWidget(title="Drawing with Points")
        self.plotWidget3.plot(np.random.normal(size=100), pen=None, symbol='o')
        self.layout.addWidget(self.plotWidget3, 1, 0)

        # Parametric plot with grid enabled
        self.plotWidget4 = pg.PlotWidget(title="Parametric Plot with Grid")
        t = np.linspace(0, 2*np.pi, 1000)
        self.plotWidget4.plot(np.sin(t), np.cos(t))
        self.plotWidget4.showGrid(x=True, y=True)
        self.layout.addWidget(self.plotWidget4, 1, 1)

        # Scatter plot with axis labels and log scale
        self.plotWidget5 = pg.PlotWidget(title="Scatter Plot with Log Scale")
        self.plotWidget5.setLogMode(x=True, y=True)
        self.plotWidget5.setLabel('left', 'Y Axis', units='A')
        self.plotWidget5.setLabel('bottom', 'X Axis', units='s')
        self.plotWidget5.plot(np.random.exponential(size=100), symbol='t', pen=None)
        self.layout.addWidget(self.plotWidget5, 2, 0)

        # Updating plot
        self.plotWidget6 = pg.PlotWidget(title="Updating Plot")
        self.curve = self.plotWidget6.plot(np.random.normal(size=100))
        self.layout.addWidget(self.plotWidget6, 2, 1)
        self.timer = QTimer()
        self.timer.timeout.connect(self.updatePlot)
        self.timer.start(1000)

        # Filled plot with axis disabled
        self.plotWidget7 = pg.PlotWidget(title="Filled Plot with Axis Disabled")
        y = np.random.normal(size=100).cumsum()
        self.plotWidget7.plot(y, fillLevel=-0.3, brush=(50, 50, 200, 100))
        self.plotWidget7.showAxis('left', False)
        self.plotWidget7.showAxis('bottom', False)
        self.layout.addWidget(self.plotWidget7, 3, 0)

        # Region selection and zoom on selected region
        self.plotWidget8 = pg.PlotWidget(title="Region Selection and Zoom")
        self.plotWidget8.plot(np.random.normal(size=100).cumsum())
        self.region = pg.LinearRegionItem()
        self.region.setZValue(10)
        self.plotWidget8.addItem(self.region)
        self.region.sigRegionChanged.connect(self.updateRegion)
        self.layout.addWidget(self.plotWidget8, 3, 1)

        self.regionPlot = pg.PlotWidget(title="Zoomed Region")
        self.layout.addWidget(self.regionPlot, 4, 0, 1, 2)
        self.updateRegion()

    def updatePlot(self):
        self.curve.setData(np.random.normal(size=100))

    def updateRegion(self):
        minX, maxX = self.region.getRegion()
        self.regionPlot.setXRange(minX, maxX, padding=0)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    main = PlottingWindow()
    main.show()
    sys.exit(app.exec_())
```