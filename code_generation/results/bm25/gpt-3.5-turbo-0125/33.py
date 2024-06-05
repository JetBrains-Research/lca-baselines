```python
import sys
from PyQt5.QtWidgets import QApplication
import pyqtgraph as pg

app = QApplication(sys.argv)

win = pg.GraphicsLayoutWidget()
win.show()

plot1 = win.addPlot()
img1 = pg.ImageItem()
plot1.addItem(img1)
colorbar1 = pg.ColorBarItem()
plot1.addItem(colorbar1)

plot2 = win.addPlot()
img2 = pg.ImageItem()
plot2.addItem(img2)
colorbar2 = pg.ColorBarItem(orientation='horizontal')
plot2.addItem(colorbar2)

plot3 = win.addPlot()
img3 = pg.ImageItem()
plot3.addItem(img3)
colorbar3 = pg.ColorBarItem()
plot3.addItem(colorbar3)

plot4 = win.addPlot()
img4 = pg.ImageItem()
plot4.addItem(img4)
colorbar4 = pg.ColorBarItem()
plot4.addItem(colorbar4)

if __name__ == '__main__':
    if sys.flags.interactive != 1:
        sys.exit(app.exec_())
```