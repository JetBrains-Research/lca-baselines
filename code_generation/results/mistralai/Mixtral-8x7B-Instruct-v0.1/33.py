 ```python
import pyqtgraph as pg
from pyqtgraph.colorbar import ColorBarItem
import numpy as np

app = pg.mkQApp()

# Create the main window
win = pg.GraphicsWindow()

# Create the first plot - non-interactive image with vertical color bar
img1 = np.random.random((10, 10))
plot1 = win.addPlot()
imgItem1 = pg.ImageItem(img1)
plot1.addItem(imgItem1)
cb1 = ColorBarItem(imageItem=imgItem1, orientation='vertical')
win.layout().addItem(cb1, 0, 1)

# Create the second plot - interactive noisy image with horizontal color bar
img2 = np.random.random((10, 10)) + np.random.normal(size=(10, 10))
plot2 = win.addPlot()
imgItem2 = pg.ImageItem(img2)
plot2.addItem(imgItem2)
cb2 = ColorBarItem(imageItem=imgItem2, orientation='horizontal')
plot2.addItem(cb2, row=1, col=0)

# Create the third and fourth plots - noisy images with shared color bar
img3 = np.random.random((10, 10)) + np.random.normal(size=(10, 10))
img4 = np.random.random((10, 10)) + np.random.normal(size=(10, 10))
shared_cb = ColorBarItem()
plot3 = win.addPlot()
plot4 = win.addPlot()
imgItem3 = pg.ImageItem(img3)
imgItem4 = pg.ImageItem(img4)
plot3.addItem(imgItem3)
plot4.addItem(imgItem4)
plot3.addItem(shared_cb, row=1, col=0)
plot4.addItem(shared_cb, row=1, col=0)
shared_cb.setImageItem(imgItem3)

win.show()

if __name__ == '__main__':
    app.exec_()
```