```python
import pyqtgraph as pg
import numpy as np
from pyqtgraph.Qt import QtGui

app = QtGui.QApplication([])
win = pg.GraphicsWindow(title="Basic plotting examples")
win.resize(1000, 600)

p1 = win.addPlot(title="Basic Array Plot")
p1.plot(np.random.normal(size=100))

p2 = win.addPlot(title="Multiple Curves Plot")
p2.plot(np.random.normal(size=100), pen=(255,0,0))
p2.plot(np.random.normal(size=100), pen=(0,255,0))
p2.plot(np.random.normal(size=100), pen=(0,0,255))

p3 = win.addPlot(title="Drawing with Points")
p3.plot(np.random.normal(size=100), symbol='o')

p4 = win.addPlot(title="Parametric Plot with Grid")
p4.plot(np.sin(np.linspace(0, 10, 100)), np.cos(np.linspace(0, 10, 100), grid=True)

p5 = win.addPlot(title="Scatter Plot with Axis Labels and Log Scale")
p5.plot(np.random.normal(size=100), np.random.normal(size=100), pen=None, symbol='o')
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
p7.plot(np.random.normal(size=100), fillLevel=0, brush=(50,50,200,100))
p7.hideAxis('bottom')
p7.hideAxis('left')

p8 = win.addPlot(title="Region Selection and Zoom")
p8.plot(np.random.normal(size=100))
region = pg.LinearRegionItem([20, 80])
region.setZValue(-10)
p8.addItem(region)
def updateRegion():
    region.setZValue(-10)
    region.setRegion([i+np.random.normal()*10 for i in region.getRegion()])
timer2 = pg.QtCore.QTimer()
timer2.timeout.connect(updateRegion)
timer2.start(500)

win.nextRow()

p9 = win.addPlot(title="Zoom on Selected Region")
p9.plot(np.random.normal(size=100))
vb = p9.getViewBox()
vb.setAspectLocked()
def updateViews():
    p9.enableAutoRange('xy', False)
    p9.autoRange()
timer3 = pg.QtCore.QTimer()
timer3.timeout.connect(updateViews)
timer3.start(100)

if __name__ == '__main__':
    QtGui.QApplication.instance().exec_()
```