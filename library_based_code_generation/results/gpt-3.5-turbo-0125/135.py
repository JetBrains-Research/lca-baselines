```python
import pyqtgraph as pg
import numpy as np
from pyqtgraph.Qt import QtGui

app = QtGui.QApplication([])
win = pg.GraphicsWindow(title="Basic plotting examples")
win.resize(1000, 600)

p1 = win.addPlot(title="Basic Array Plot")
p1.plot(np.random.normal(size=100))

p2 = win.addPlot(title="Multiple Curves")
p2.plot(np.random.normal(size=100), pen=(255,0,0))
p2.plot(np.random.normal(size=100), pen=(0,255,0))
p2.plot(np.random.normal(size=100), pen=(0,0,255))

p3 = win.addPlot(title="Drawing with Points")
p3.plot(np.random.normal(size=100), symbol='o')

p4 = win.addPlot(title="Parametric Plot with Grid")
x = np.linspace(-2*np.pi, 2*np.pi, 100)
y = np.sin(x)
p4.plot(x, y, grid=True)

p5 = win.addPlot(title="Scatter Plot with Axis Labels and Log Scale")
x = np.random.normal(size=1000)
y = np.random.normal(size=1000)
p5.plot(x, y, pen=None, symbol='o')
p5.setLabel('left', "Y Axis Label")
p5.setLabel('bottom', "X Axis Label")
p5.setLogMode(x=True, y=True)

p6 = win.addPlot(title="Updating Plot")
curve = p6.plot(pen='y')
data = np.random.normal(size=100)
def update():
    data[:-1] = data[1:]
    data[-1] = np.random.normal()
    curve.setData(data)
timer = pg.QtCore.QTimer()
timer.timeout.connect(update)
timer.start(50)

p7 = win.addPlot(title="Filled Plot with Axis Disabled")
x = np.linspace(-2*np.pi, 2*np.pi, 100)
y1 = np.sin(x)
y2 = np.cos(x)
p7.plot(x, y1, fillLevel=0, brush=(50,50,200,100))
p7.plot(x, y2, fillLevel=0, brush=(50,50,200,100))
p7.showAxis('bottom', False)
p7.showAxis('left', False)

p8 = win.addPlot(title="Region Selection and Zoom")
x = np.linspace(-2*np.pi, 2*np.pi, 100)
y = np.sin(x)
p8.plot(x, y)
region = pg.LinearRegionItem([20, 40])
p8.addItem(region)
p8.setAutoVisible(y=True)

win.nextRow()

win.show()
app.exec_()
```