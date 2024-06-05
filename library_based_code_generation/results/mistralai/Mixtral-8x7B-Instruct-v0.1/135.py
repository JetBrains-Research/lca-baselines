 ```python
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtWidgets
import numpy as np

app = QtWidgets.QApplication([])

win = pg.GraphicsWindow(title="Basic plotting examples")
win.setWindowTitle('Basic plotting examples')
win.setGeometry(100, 100, 1200, 800)

layout = win.addLayout(row=0, col=0)

# Basic array plotting
plot1 = win.addPlot(title="Basic array plotting")
plot1.plot(np.random.normal(size=100))

# Multiple curves
plot2 = win.addPlot(title="Multiple curves")
data2 = [np.random.normal() for _ in range(100)]
plot2.plot(data2, pen='r', name='Red curve')
plot2.plot(data2, pen='g', name='Green curve')

# Drawing with points
plot3 = win.addPlot(title="Drawing with points")
plot3.setMouseEnabled(x=False, y=False)
plot3.scene().sigMouseClicked.connect(plot3.plot)

# Parametric plot with grid enabled
plot4 = win.addPlot(title="Parametric plot with grid enabled")
plot4.setGrid(True)
curve4 = pg.ParametricCurve(pen='b')
plot4.addItem(curve4)

# Scatter plot with axis labels and log scale
plot5 = win.addPlot(title="Scatter plot with axis labels and log scale")
plot5.setLabel('left', 'Y-axis')
plot5.setLabel('bottom', 'X-axis')
plot5.getAxis('bottom').setScale(10**np.linspace(0, 3, 11))
plot5.plot(np.random.normal(size=100), np.random.normal(size=100), pen='r', symbol='o')

# Updating plot
def update_plot():
    data6[0] += np.random.normal()
    data6[1] += np.random.normal()
    plot6.setData(data6)

data6 = [np.random.normal(size=100), np.random.normal(size=100)]
plot6 = win.addPlot(title="Updating plot")
curve6 = plot6.plot(data6[0], data6[1], pen='r')
timer = QtCore.QTimer()
timer.timeout.connect(update_plot)
timer.start(100)

# Filled plot with axis disabled
plot7 = win.addPlot(title="Filled plot with axis disabled")
plot7.setAxisVisible(False, False)
curve7 = pg.Curve(pen='r')
curve7.setSymbol('o')
plot7.addItem(curve7)

# Region selection
plot8 = win.addPlot(title="Region selection")
plot8.setMouseEnabled(x=True, y=False)
plot8.scene().sigMouseMoved.connect(lambda: print(plot8.vb.viewBox().viewRange()))

# Zoom on selected region
plot9 = win.addPlot(title="Zoom on selected region")
plot9.setRange(xRange=[0, 100], yRange=[-5, 5])
plot9.setMouseEnabled(x=True, y=True)
plot9.scene().sigMouseClicked.connect(lambda pos: plot9.setRange(xRange=[pos.x(), pos.x() + 10], yRange=[0, 10]))

win.show()

if __name__ == '__main__':
    import sys
    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        QtWidgets.QApplication.instance().exec_()
```