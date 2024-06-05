 ```python
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui

app = QtGui.QApplication.instance()
if app is None:
    app = QtGui.QApplication([])

window = pg.GraphicsLayoutWidget()
window.setWindowTitle('PyQtGraph Example')

# Plot 1: Non-interactive image with vertical color bar
image1 = pg.ImageItem()
color_bar1 = pg.ColorBarItem(image=image1)
plot1 = window.addPlot()
plot1.addItem(image1)
plot1.addItem(color_bar1)
plot1.setMouseEnabled(False, False)

# Plot 2: Interactive noisy image with horizontal color bar
image2 = pg.ImageItem()
color_bar2 = pg.ColorBarItem(orientation='bottom', image=image2)
plot2 = window.addPlot()
plot2.addItem(image2)
plot2.addItem(color_bar2)

# Plots 3 and 4: Interactive noisy images with shared color bar
image3 = pg.ImageItem()
image4 = pg.ImageItem()
color_bar3 = pg.ColorBarItem(image=image3)
plot3 = window.addPlot()
plot3.addItem(image3)
plot3.addItem(color_bar3)
plot4 = window.addPlot()
plot4.addItem(image4)
plot4.addItem(color_bar3)

window.show()

if app.isSingleInstance():
    app.exec_()
```