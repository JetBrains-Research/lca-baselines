 Here is the Python code for the PyQtGraph application as per your instructions:

```python
import sys
from PyQtGraph.Qt import QtGui, QtCore
from PyQtGraph.GraphicsLayout import GraphicsLayoutWidget
from PyQtGraph.ImageItem import ImageItem
from PyQtGraph.ColorBarItem import ColorBarItem
from PyQtGraph.ScaleBar import ScaleBar

class MainWindow(QtGui.QMainWindow):
    def __init__(self):
        super().__init__()

        self.gl = GraphicsLayoutWidget()
        self.setCentralWidget(self.gl)

        # Non-interactive plot with vertical color bar
        plot1 = self.gl.addPlot(row=0, col=0, colspan=2)
        image1 = ImageItem(np.random.rand(100, 100), autoLoc=False)
        color_bar1 = ColorBarItem(orientation='vt', size=0.1, color=QtGui.QColor(255, 0, 0), parent=plot1)
        plot1.addItem(image1)
        plot1.addItem(color_bar1)

        # Interactive plot with horizontal color bar
        plot2 = self.gl.addPlot(row=1, col=0)
        image2 = ImageItem(np.random.rand(100, 100) + np.random.normal(0, 0.1, size=(100, 100)), autoLoc=False)
        color_bar2 = ColorBarItem(orientation='hl', size=0.1, color=QtGui.QColor(0, 255, 0), parent=plot2)
        plot2.addItem(image2)
        plot2.addItem(color_bar2)

        # Shared color bar for plots 3 and 4
        color_bar3 = ColorBarItem(orientation='hl', size=0.1, color=QtGui.QColor(0, 0, 255), parent=self.gl)

        # Plots 3 and 4 with shared color bar
        plot3 = self.gl.addPlot(row=1, col=1)
        image3 = ImageItem(np.random.rand(100, 100) + np.random.normal(0, 0.1, size=(100, 100)), autoLoc=False)
        plot3.addItem(image3)
        plot3.addItem(color_bar3)

        plot4 = self.gl.addPlot(row=2, col=1)
        image4 = ImageItem(np.random.rand(100, 100) + np.random.normal(0, 0.1, size=(100, 100)), autoLoc=False)
        plot4.addItem(image4)
        plot4.addItem(color_bar3)

        self.show()

if __name__ == "__main__":
    app = QtGui.QApplication(sys.argv)
    window = MainWindow()
    sys.exit(app.exec_())
```

This code creates a PyQtGraph application with a main window that displays three plots as described in the instructions. The first plot is non-interactive and displays an image with a vertical color bar, the second plot is interactive and displays a noisy image with a horizontal color bar, and the third and fourth plots display noisy images and share a separate color bar. The color bars are created using the ColorBarItem class from the PyQtGraph library. The images are created using the ImageItem class from the PyQtGraph library, and the plots are created using the addPlot method of a GraphicsLayoutWidget instance. The application starts the Qt event loop if it is the main module.