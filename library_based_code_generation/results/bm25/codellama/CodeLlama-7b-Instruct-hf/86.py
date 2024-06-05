  import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui
import numpy as np

class ScatterPlotDemo(QtGui.QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setWindowTitle('Scatter Plot Demo')
        self.setGeometry(300, 300, 800, 600)

        self.layout = pg.GraphicsLayoutWidget()
        self.setCentralWidget(self.layout)

        self.plots = []
        self.plot_data = []

        self.create_plots()

    def create_plots(self):
        self.plots.append(self.create_identical_transform_invariant_plot())
        self.plots.append(self.create_transform_invariant_plot())
        self.plots.append(self.create_non_transform_invariant_plot())
        self.plots.append(self.create_large_scatter_plot())

        for plot in self.plots:
            self.layout.addPlot(plot)

    def create_identical_transform_invariant_plot(self):
        plot = pg.ScatterPlot()
        plot.setTitle('Identical and Transform-Invariant')
        plot.setLabel('left', 'X')
        plot.setLabel('bottom', 'Y')
        plot.setAspectLocked(True)

        data = np.random.rand(100, 2)
        plot.plot(data[:, 0], data[:, 1], symbol='o', symbolSize=10, symbolPen=None, symbolBrush=(255, 0, 0))

        return plot

    def create_transform_invariant_plot(self):
        plot = pg.ScatterPlot()
        plot.setTitle('Transform-Invariant')
        plot.setLabel('left', 'X')
        plot.setLabel('bottom', 'Y')
        plot.setAspectLocked(True)

        data = np.random.rand(100, 2)
        plot.plot(data[:, 0], data[:, 1], symbol='o', symbolSize=10, symbolPen=None, symbolBrush=(0, 255, 0))

        return plot

    def create_non_transform_invariant_plot(self):
        plot = pg.ScatterPlot()
        plot.setTitle('Non-Transform-Invariant')
        plot.setLabel('left', 'X')
        plot.setLabel('bottom', 'Y')
        plot.setAspectLocked(True)

        data = np.random.rand(100, 2)
        plot.plot(data[:, 0], data[:, 1], symbol='o', symbolSize=10, symbolPen=None, symbolBrush=(0, 0, 255))

        return plot

    def create_large_scatter_plot(self):
        plot = pg.ScatterPlot()
        plot.setTitle('Large Scatter Plot')
        plot.setLabel('left', 'X')
        plot.setLabel('bottom', 'Y')
        plot.setAspectLocked(True)

        data = np.random.rand(10000, 2)
        plot.plot(data[:, 0], data[:, 1], symbol='o', symbolSize=10, symbolPen=None, symbolBrush=(255, 255, 0))

        return plot

    def plotClicked(self, plot, points):
        for point in points:
            plot.plot(point.x(), point.y(), symbol='o', symbolSize=10, symbolPen=None, symbolBrush=(255, 0, 0))

if __name__ == '__main__':
    app = QtGui.QApplication([])
    demo = ScatterPlotDemo()
    demo.show()
    app.exec_()