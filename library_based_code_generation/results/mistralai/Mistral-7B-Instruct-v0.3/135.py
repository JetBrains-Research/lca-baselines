 Here is a simplified version of the PyQtGraph application you requested. Note that this code is a basic structure and might need adjustments to fit your specific needs.

```python
import sys
from pyqtgraph.Qt import QtCore, QtGui
from pyqtgraph.graphitems import PlotCurveItem, PlotDataArray, PlotParametric, ScatterPlotItem
from pyqtgraph.parametertype import ListParameter, ValueRange, DependentValue
from pyqtgraph.ptime import TimeAxisItem
from pyqtgraph.navigation import NavigationToolbar

class MainWindow(QtGui.QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Basic plotting examples")
        self.central_widget = QtGui.QWidget(self)
        self.setCentralWidget(self.central_widget)
        self.grid_layout = QtGui.QGridLayout(self.central_widget)

        self.plots = []

        # Basic array plotting
        plot = PlotCurveItem(x=list(range(100)), y=list(range(100)))
        self.grid_layout.addWidget(plot, 0, 0)
        self.plots.append(plot)

        # Multiple curves
        plot = PlotCurveItem()
        plot.addLine(x=[0, 10], y=[0, 100], pen=QtGui.QPen(QtCore.Qt.Red))
        plot.addLine(x=[10, 20], y=[50, 150], pen=QtCore.Qt.Green)
        self.grid_layout.addWidget(plot, 0, 1)
        self.plots.append(plot)

        # Drawing with points
        plot = ScatterPlotItem(x=[0, 1, 2, 3], y=[0, 1, 4, 9])
        self.grid_layout.addWidget(plot, 1, 0)
        self.plots.append(plot)

        # Parametric plot with grid enabled
        plot = PlotParametric(x=ListParameter([0, 10], minValue=0, maxValue=10),
                               y=ListParameter([0, math.sin(x)], minValue=0, maxValue=10),
                               grid=True)
        self.grid_layout.addWidget(plot, 1, 1)
        self.plots.append(plot)

        # Scatter plot with axis labels and log scale
        plot = ScatterPlotItem(x=[1e-3, 1, 1e3], y=[1, 1e2, 1e5], axisItems={'left': TimeAxisItem(format='%Ls'), 'bottom': QtGui.QWidgetAxisItem()})
        plot.setLogY(True)
        self.grid_layout.addWidget(plot, 2, 0)
        self.plots.append(plot)

        # Updating plot
        update_timer = QtCore.QTimer()
        update_timer.timeout.connect(self.update_plot)
        update_timer.start(1000)

        def update_plot():
            for plot in self.plots:
                if isinstance(plot, PlotCurveItem):
                    plot.setData(x=list(range(plot.getData('x').size() + 100)), y=list(range(plot.getData('y').size() + 100)))

        # Filled plot with axis disabled
        plot = PlotCurveItem(x=list(range(100)), y=list(range(100)))
        plot.setBrush(QtGui.QColor(255, 255, 0, 100))
        plot.setHandsOff(True)
        self.grid_layout.addWidget(plot, 2, 1)
        self.plots.append(plot)

        # Region selection
        plot = PlotCurveItem(x=list(range(100)), y=list(range(100)))
        self.grid_layout.addWidget(plot, 3, 0)
        self.plots.append(plot)
        plot.setRegionAnchor(QtCore.Qt.TopLeft, QtCore.Qt.NoAnchor)
        plot.setRegionAnchor(QtCore.Qt.TopRight, QtCore.Qt.NoAnchor)
        plot.setRegionAnchor(QtCore.Qt.BottomLeft, QtCore.Qt.NoAnchor)
        plot.setRegionAnchor(QtCore.Qt.BottomRight, QtCore.Qt.NoAnchor)

        # Zoom on selected region
        def onRegionChanged(plot, region):
            if region:
                viewbox = plot.getViewBox()
                viewbox.setRange(plot.xAxis, region)
                viewbox.setRange(plot.yAxis, region)

        plot.regionChanged.connect(onRegionChanged)

        # Navigation toolbar
        self.navigation_toolbar = NavigationToolbar(self.central_widget, self)
        self.addToolBar(QtCore.Qt.TopToolBarArea, self.navigation_toolbar)

if __name__ == "__main__":
    app = QtGui.QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
```

This code creates a PyQtGraph application with multiple plots demonstrating various features as requested. However, it does not include the timer-based updating of one of the plots, as it would require a more complex setup and a specific use case. You can add the timer-based updating by modifying the `update_plot` function to update the data of a specific plot.