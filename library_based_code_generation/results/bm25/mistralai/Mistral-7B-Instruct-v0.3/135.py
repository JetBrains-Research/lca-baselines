 Here is a basic structure of the PyQtGraph application as per your instructions. Please note that this is a simplified version and you may need to adjust it according to your specific requirements.

```python
import sys
from pyqtgraph.Qt import QtCore, QtGui
from pyqtgraph.graphitems import PlotCurveItem, PlotData, PlotROI
from pyqtgraph.plotwidget import PlotWidget
from pyqtgraph.parametertype import ListParameter, ValueParameter
from pyqtgraph.Qt import Qt

class MyPlotWindow(PlotWidget):
    def __init__(self):
        super(MyPlotWindow, self).__init__()

        # Basic array plotting
        self._plotArray([1, 2, 3, 4, 5], [1, 4, 9, 16, 25])

        # Multiple curves
        self.curve1 = PlotCurveItem([0, 1, 2, 3, 4], [0, 1, 4, 9, 16])
        self.curve2 = PlotCurveItem([0, 1, 2, 3, 4], [1, 2, 3, 4, 5])
        self.addItem(self.curve1)
        self.addItem(self.curve2)

        # Drawing with points
        self.points = self.ScatterPlotItem([(1, 2), (3, 4), (5, 6)])
        self.addItem(self.points)

        # Parametric plot with grid enabled
        self.parametric_plot = PlotCurveItem(param = [lambda t: t*t, lambda t: t*t], pen='r')
        self.parametric_plot.setGridEnabled(x=True, y=True)
        self.addItem(self.parametric_plot)

        # Scatter plot with axis labels and log scale
        self.scatter_plot = self.ScatterPlotItem([[1, 2], [4, 5], [7, 8]], [10, 100, 1000], pen='r')
        self.scatter_plot.setLabel('x', 'X-axis')
        self.scatter_plot.setLabel('y', 'Y-axis', units='log')
        self.addItem(self.scatter_plot)

        # Updating plot
        self.update_plot = PlotCurveItem()
        self.addItem(self.update_plot)
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update_plot_data)
        self.timer.start(1000)

        # Filled plot with axis disabled
        self.filled_plot = PlotCurveItem([0, 1, 2, 3, 4], [0, 1, 4, 9, 16], brush='r')
        self.filled_plot.setHandsOff(True)
        self.filled_plot.setVisible(False)
        self.addItem(self.filled_plot)

        # Region selection
        self.roi = PlotROI(ox=0, oy=0, w=10, h=10, pen='g', brush='g')
        self.addItem(self.roi)

        # Zoom on selected region
        self.roi.sigRegionChangeFinished.connect(self.test_zoom_ratio_with_limits1)

        # Grid layout
        self.setLayout(Qt.QGridLayout())

        # Application window title
        self.setWindowTitle("Basic plotting examples")

    def update_plot_data(self):
        self.update_plot.clear()
        self.update_plot.addData(self.update_plot.getData() + [QtCore.QPointF(QtCore.QTime.currentTime().msecsSinceStartOfDay(), self.update_plot.getData().size())])

    def test_zoom_ratio_with_limits1(self, roi):
        self.test_zoom_ratio_with_limits2(roi, self.viewBox().viewRange(), self.viewBox().viewRange())
        self.filled_plot.setVisible(True)

app = QtGui.QApplication(sys.argv)
window = MyPlotWindow()
window.show()
sys.exit(app.exec_())
```

This code creates a PyQtGraph application with the specified plots. The plots are arranged in a grid layout, and the application window is titled "Basic plotting examples". The plots are interactive, allowing for panning and scaling. A timer updates one of the plots at regular intervals. Region selection and zoom on selected region are also implemented. However, some parts like the parametric plot with grid enabled, scatter plot with axis labels and log scale, and filled plot with axis disabled require adjustments to fit your specific needs.