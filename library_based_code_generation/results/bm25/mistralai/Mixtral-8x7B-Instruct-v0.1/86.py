 ```python
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui
import numpy as np

class ScatterPlotApp:
    def __init__(self):
        self.app = QtGui.QApplication([])

        self.main_window = QtGui.QMainWindow()
        self.main_widget = QtGui.QWidget()
        self.main_layout = QtGui.QVBoxLayout()
        self.main_widget.setLayout(self.main_layout)
        self.main_window.setCentralWidget(self.main_widget)

        self.multi_plot_widget = pg.MultiPlotWidget()
        self.main_layout.addWidget(self.multi_plot_widget)

        self.plot_data_1 = self.generate_random_data(100)
        self.plot_1 = self.multi_plot_widget.addPlot()
        self.scatter_plot_1 = pg.ScatterPlotItem(self.plot_data_1, size=10, pen='r', brush='r')
        self.plot_1.addItem(self.scatter_plot_1)

        self.plot_data_2 = self.generate_random_data(100, unique_colors=True)
        self.plot_2 = self.multi_plot_widget.addPlot()
        self.scatter_plot_2 = pg.ScatterPlotItem(self.plot_data_2, size=10, pen='b', brush='b')
        self.plot_2.addItem(self.scatter_plot_2)

        self.plot_data_3 = self.generate_random_data(100, unique_colors=True, transform_invariant=False)
        self.plot_3 = self.multi_plot_widget.addPlot()
        self.scatter_plot_3 = pg.ScatterPlotItem(self.plot_data_3, size=10, pen='g', brush='g')
        self.plot_3.addItem(self.scatter_plot_3)

        self.plot_data_4 = self.generate_large_random_data(10000)
        self.plot_4 = self.multi_plot_widget.addPlot()
        self.scatter_plot_4 = pg.ScatterPlotItem(self.plot_data_4, size=5, pen='k', brush='k')
        self.plot_4.addItem(self.scatter_plot_4)

        self.scatter_plot_1.sigClicked.connect(lambda: self.highlight_clicked_points(self.scatter_plot_1))
        self.scatter_plot_2.sigClicked.connect(lambda: self.highlight_clicked_points(self.scatter_plot_2))
        self.scatter_plot_3.sigClicked.connect(lambda: self.highlight_clicked_points(self.scatter_plot_3))
        self.scatter_plot_4.sigClicked.connect(lambda: self.highlight_clicked_points(self.scatter_plot_4))

    def generate_random_data(self, size, unique_colors=False, transform_invariant=True):
        data = np.random.rand(size, 2)
        if not transform_invariant:
            data += np.random.rand(size, 2)
        colors = np.random.rand(size, 3) if unique_colors else np.array([[1, 0, 0]] * size)
        return pg.PlotDataItem(data, pen=colors, symbol='o')

    def generate_large_random_data(self, size):
        return self.generate_random_data(size, unique_colors=False, transform_invariant=False)

    def highlight_clicked_points(self, scatter_plot):
        clicked_points = scatter_plot.pointsAt(scatter_plot.getViewBox().mapSceneToView(QtGui.QCursor.pos()))
        for point in clicked_points:
            scatter_plot.setBrush(point, pg.mkBrush(255, 255, 0))
        scatter_plot.updateItems()

    def run(self):
        self.main_window.show()
        self.app.exec_()

if __name__ == "__main__":
    app = ScatterPlotApp()
    app.run()
```