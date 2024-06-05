```python
import sys
import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtGui

app = QtGui.QApplication([])
win = pg.GraphicsLayoutWidget(show=True, title="Scatter Plot Demo")

plot1 = win.addPlot(title="Identical and Transform-Invariant Spots")
plot1.setData(pos=np.random.normal(size=(100, 2), scale=0.1))

plot2 = win.addPlot(title="Transform-Invariant, Non-Identical Spots")
plot2.setData(pos=np.random.normal(size=(100, 2), scale=[0.1, 0.2]))

plot3 = win.addPlot(title="Non-Identical and Non-Transform-Invariant Spots")
plot3.setData(pos=np.random.normal(size=(100, 2), scale=[0.1, 0.2]))

plot4 = win.addPlot(title="Performance Test for Large Scatterplots")
plot4.setData(pos=np.random.normal(size=(10000, 2), scale=0.1))

def clicked(plot, points):
    print(f"Clicked points: {points}")

plot1.scatter.sigClicked.connect(lambda points: clicked(plot1, points))
plot2.scatter.sigClicked.connect(lambda points: clicked(plot2, points))
plot3.scatter.sigClicked.connect(lambda points: clicked(plot3, points))
plot4.scatter.sigClicked.connect(lambda points: clicked(plot4, points))

if __name__ == '__main__':
    if sys.flags.interactive != 1 or not hasattr(QtCore, 'PYQT_VERSION'):
        QtGui.QApplication.instance().exec_()
```