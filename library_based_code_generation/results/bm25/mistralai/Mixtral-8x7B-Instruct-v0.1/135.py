 ```python
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtWidgets
import numpy as np

app = QtWidgets.QApplication([])

main_window = QtWidgets.QMainWindow()
main_window.setWindowTitle('Basic plotting examples')

central_widget = QtWidgets.QWidget()
main_window.setCentralWidget(central_widget)

layout = QtWidgets.QGridLayout(central_widget)

plot_area_1 = pg.PlotWidget()
layout.addWidget(plot_area_1, 0, 0)
plot_area_1.clearPlots()
plot_area_1.plot(_plotArray(np.random.normal(size=(30, 2)), pen='b'))
plot_area_1.plot(_plotArray(np.random.normal(size=(30, 2)), pen='r'))

plot_area_2 = pg.PlotWidget()
layout.addWidget(plot_area_2, 0, 1)
plot_area_2.clearPlots()
plot_area_2.plot(_plotArray(np.random.normal(size=(30, 2)), pen='g'))
plot_area_2.plot(_plotArray(np.random.normal(size=(30, 2)), pen='y'))

plot_area_3 = pg.PlotWidget()
layout.addWidget(plot_area_3, 1, 0)
plot_area_3.clearPlots()
plot_area_3.setMouseEnabled(x=False, y=False)
plot_area_3.plot(np.sin, pen='b')
plot_area_3.plot(np.cos, pen='r')

plot_area_4 = pg.PlotWidget()
layout.addWidget(plot_area_4, 1, 1)
plot_area_4.clearPlots()
plot_area_4.setLabel('left', 'Y-axis')
plot_area_4.setLabel('bottom', 'X-axis')
plot_area_4.setLogMode(x=False, y=True)
plot_area_4.scatterPlot(_plotArray(np.random.normal(size=(30, 2)), pen='b'), symbol='o')

plot_area_5 = pg.PlotWidget()
layout.addWidget(plot_area_5, 2, 0)
plot_area_5.clearPlots()
plot_area_5.enableAutoRange(x=False, y=False)
plot_area_5.setXRange(0, 10, padding=0)
plot_area_5.setYRange(0, 10, padding=0)
plot_area_5.plot(_plotArray(np.random.normal(size=(30, 2)), pen='b', fillLevel=0, fillBrush=(255, 255, 0, 128)))

plot_area_6 = pg.PlotWidget()
layout.addWidget(plot_area_6, 2, 1)
plot_area_6.clearPlots()
plot_area_6.setMouseEnabled(x=False, y=False)
plot_area_6.setLabel('left', 'Y-axis')
plot_area_6.setLabel('bottom', 'X-axis')
plot_area_6.addItem(pg.RectROI([2, 4], [4, 6], pen=(255, 0, 0, 255), fillBrush=(255, 0, 0, 128)))

plot_area_7 = pg.PlotWidget()
layout.addWidget(plot_area_7, 3, 0)
plot_area_7.clearPlots()
plot_area_7.setMouseEnabled(x=False, y=False)
plot_area_7.setXRange(0, 10, padding=0)
plot_area_7.setYRange(0, 10, padding=0)
plot_area_7.setLimits(xMin=0, xMax=10, yMin=0, yMax=10)

def update_plot():
    plot_area_7.plot(_plotArray(np.random.normal(size=(30, 2)), pen='b'), clear=True)

timer = QtCore.QTimer()
timer.timeout.connect(update_plot)
timer.start(1000)

main_window.show()
app.exec_()
```