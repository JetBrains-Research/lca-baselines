 ```python
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtWidgets
import random

class ScatterPlotApp(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()

        self.initUI()

    def initUI(self):
        central_widget = QtWidgets.QWidget()
        self.setCentralWidget(central_widget)

        layout = QtWidgets.QHBoxLayout()
        central_widget.setLayout(layout)

        self.plot_widget = pg.GraphicsLayoutWidget()
        layout.addWidget(self.plot_widget)

        self.create_plots()

        self.show()

    def create_plots(self):
        self.data_1 = [random.randint(0, 100) for _ in range(100)]
        self.plot_1 = self.plot_widget.addPlot()
        self.plot_1.scatterPlot(self.data_1, self.data_1, symbol='o', size=10, pen=pg.mkPen(color=(255, 0, 0)))
        self.plot_1.plotItem.scene().sigMouseClicked.connect(lambda pos, btn: self.highlight_point(pos, self.plot_1))

        self.data_2 = [random.randint(0, 100) for _ in range(100)]
        self.plot_2 = self.plot_widget.addPlot()
        self.plot_2.scatterPlot(self.data_2, self.data_2, symbol='o', size=10, pen=pg.mkPen(color=(0, 255, 0)), symbolBrush=(0, 255, 0))
        self.plot_2.plotItem.scene().sigMouseClicked.connect(lambda pos, btn: self.highlight_point(pos, self.plot_2))

        self.data_3 = [random.randint(0, 100) for _ in range(100)]
        self.plot_3 = self.plot_widget.addPlot()
        self.plot_3.scatterPlot(self.data_3, self.data_3, symbol='o', size=10, pen=pg.mkPen(color=(0, 0, 255)), symbolBrush=(0, 0, 255))
        self.plot_3.plotItem.scene().sigMouseClicked.connect(lambda pos, btn: self.highlight_point(pos, self.plot_3))

        self.data_4 = [random.randint(0, 100) for _ in range(10000)]
        self.plot_4 = self.plot_widget.addPlot()
        self.plot_4.scatterPlot(self.data_4, self.data_4, symbol='o', size=10, pen=pg.mkPen(color=(255, 255, 0)), symbolBrush=(255, 255, 0))
        self.plot_4.plotItem.scene().sigMouseClicked.connect(lambda pos, btn: self.highlight_point(pos, self.plot_4))

    def highlight_point(self, pos, plot):
        for item in plot.items:
            if isinstance(item, pg.ScatterPlotItem):
                points = item.pointsAt(pos)
                if points:
                    point = points[0]
                    point.setPen(pg.mkPen(color=(0, 0, 0), width=3))
                else:
                    for point in item.points():
                        point.setPen(pg.mkPen(color=(255, 255, 255), width=1))

if __name__ == '__main__':
    app = QtWidgets.QApplication([])
    window = ScatterPlotApp()
    app.exec_()
```