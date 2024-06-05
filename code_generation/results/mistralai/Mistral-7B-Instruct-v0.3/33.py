 Here is the Python code for the PyQtGraph application as per your instructions:

```python
import sys
from PyQt5.QtWidgets import QApplication
from pyqtgraph.Qt import QtCore, QtGui
from pyqtgraph import GraphicsLayoutWidget, PlotWidget, ImageItem, ColorBarItem

class MainWindow(GraphicsLayoutWidget):
    def __init__(self):
        super().__init__(show=False)

        # Non-interactive plot with image and vertical color bar
        plot1 = PlotWidget(shareY=True, parent=self)
        img1 = ImageItem(np.random.rand(100, 100), autoLoc=True)
        color_bar1 = ColorBarItem(Orientation.VERTICAL, parent=plot1)
        plot1.addItem(img1)
        plot1.addItem(color_bar1)
        self.addPlot(plot1, row=0, col=0)

        # Interactive plot with image and horizontal color bar
        plot2 = PlotWidget(shareY=True, parent=self)
        img2 = ImageItem(np.random.rand(100, 100) + np.random.normal(0, 0.1, size=(100, 100)), autoLoc=True)
        color_bar2 = ColorBarItem(Orientation.HORIZONTAL, parent=plot2)
        plot2.addItem(img2)
        plot2.addItem(color_bar2)
        self.addPlot(plot2, row=0, col=1)

        # Shared color bar for the third and fourth plots
        color_bar3 = color_bar2.clone()

        # Interactive plots with noisy images
        plot3 = PlotWidget(shareY=True, parent=self)
        img3 = ImageItem(np.random.rand(100, 100) + np.random.normal(0, 0.1, size=(100, 100)), autoLoc=True)
        plot3.addItem(img3)
        plot3.setTitle("Plot 3")
        self.addPlot(plot3, row=1, col=0)

        plot4 = PlotWidget(shareY=True, parent=self)
        img4 = ImageItem(np.random.rand(100, 100) + np.random.normal(0, 0.1, size=(100, 100)), autoLoc=True)
        plot4.addItem(img4)
        plot4.setTitle("Plot 4")
        self.addPlot(plot4, row=1, col=1)

        # Add shared color bar to the third and fourth plots
        plot3.addItem(color_bar3)
        plot4.addItem(color_bar3)

        self.show()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    sys.exit(app.exec_())
```

This code creates a PyQtGraph application with a main window that displays three plots as described in the instructions. The images are created using the ImageItem class, and the color bars are created using the ColorBarItem class. The plots are created using the addPlot method of a GraphicsLayoutWidget instance. The application starts the Qt event loop if it is the main module.