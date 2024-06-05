```python
import sys
import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtGui

app = QtGui.QApplication([])
win = pg.GraphicsLayoutWidget(show=True, title="Scatter Plot Demo")

plot1 = win.addPlot(title="Identical and Transform-Invariant")
plot1.plot(np.random.normal(size=100), np.random.normal(size=100), pen=None, symbol='o')

plot2 = win.addPlot(title="Transform-Invariant, Not Identical")
plot2.plot(np.random.normal(size=100), np.random.normal(size=100), pen=None, symbol='x')

plot3 = win.addPlot(title="Not Transform-Invariant, Not Identical")
plot3.plot(np.random.normal(size=100), np.random.normal(size=100), pen=None, symbol='t')

plot4 = win.addPlot(title="Performance Test")
plot4.plot(np.random.normal(size=10000), np.random.normal(size=10000), pen=None, symbol='o')

def clicked(plot, points):
    for p in points:
        print(f"Clicked point: {p.pos()}")

plot1.scatterPlotWidget.sigScatterPlotClicked.connect(clicked)
plot2.scatterPlotWidget.sigScatterPlotClicked.connect(clicked)
plot3.scatterPlotWidget.sigScatterPlotClicked.connect(clicked)
plot4.scatterPlotWidget.sigScatterPlotClicked.connect(clicked)

if __name__ == '__main__':
    if sys.flags.interactive != 1 or not hasattr(QtCore, 'PYQT_VERSION'):
        QtGui.QApplication.instance().exec_()
```