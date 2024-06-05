 Here is a basic example of how you might create a scatter plot using pyqtgraph that meets your requirements. Please note that this is a simplified version and you may need to adjust it according to your specific needs.

```python
import sys
from pyqtgraph.Qt import QtCore, QtGui
from pyqtgraph.graphicsItems import PlotDataItem, PlotCurveItem
import numpy as np

class ScatterPlot(QtGui.QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle('Scatter Plot')
        self.setGeometry(300, 300, 1200, 800)

        self.view = PlotWidget()
        self.view.setBackground('w')
        self.view.setRange(xMin=-10, xMax=10, yMin=-10, yMax=10)

        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.view)
        widget = QtWidgets.QWidget()
        widget.setLayout(layout)
        self.setCentralWidget(widget)

        self.data = []
        self.highlighted_points = []

        self.create_plots()

    def create_plots(self):
        # Transform-invariant and identical points
        x = np.random.uniform(-10, 10, 100)
        y = x
        item = PlotDataItem(x, y, pen='b', brush='r')
        self.view.addItem(item)
        self.data.append((x, y))

        # Transform-invariant, but not identical points
        x = np.random.uniform(-10, 10, 100)
        y = np.sin(x)
        item = PlotDataItem(x, y, pen='g', brush='y')
        self.view.addItem(item)
        self.data.append((x, y))

        # Transform-invariant and not identical points
        x = np.random.uniform(-10, 10, 100)
        y = np.random.uniform(-10, 10, 100)
        item = PlotDataItem(x, y, pen='r', brush='b')
        self.view.addItem(item)
        self.data.append((x, y))

        # Performance test of large scatterplots
        x = np.random.uniform(-1000, 1000, 100000)
        y = np.random.uniform(-1000, 1000, 100000)
        item = PlotDataItem(x, y, pen='k', brush='w')
        self.view.addItem(item)
        self.data.append((x, y))

        self.view.sigClicked.connect(self.on_point_clicked)

    def on_point_clicked(self, x, y):
        if len(self.highlighted_points) < 10:
            self.highlighted_points.append((x, y))
            for i, (_, _x, _y) in enumerate(self.data):
                for j, (x_, y_) in enumerate(zip(_x, _y)):
                    if (x_ - x)**2 + (y_ - y)**2 < 100:
                        self.data[i][2] = QtGui.QColor(QtCore.Qt.yellow)
                        self.data[i][3] = QtCore.Qt.SolidLine

    if __name__ == '__main__':
        app = QtWidgets.QApplication(sys.argv)
        window = ScatterPlot()
        window.show()
        sys.exit(app.exec_())
```

This code creates a main window with a graphics layout widget and four scatter plots. Each plot demonstrates a different way of drawing scatter plots as specified. The plots are clickable, and the clicked points are highlighted. The application is executed if the script is run as the main program. The data for the plots is generated randomly.