```python
import numpy as np
import pyqtgraph as pg
from PyQt5 import QtWidgets
from pyqtgraph import ImageItem, ColorBarItem

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()

        # Create the main widget for the window
        central_widget = QtWidgets.QWidget()
        self.setCentralWidget(central_widget)

        # Create a layout for the central widget
        layout = QtWidgets.QVBoxLayout(central_widget)

        # Create a GraphicsLayoutWidget instance
        graphics_widget = pg.GraphicsLayoutWidget()
        layout.addWidget(graphics_widget)

        # First plot (non-interactive)
        plot1 = graphics_widget.addPlot()
        plot1.disableAutoRange()
        plot1.setMouseEnabled(x=False, y=False)
        img1 = np.random.normal(size=(100, 100))
        image_item1 = ImageItem(img1)
        plot1.addItem(image_item1)
        color_bar1 = ColorBarItem(values=(img1.min(), img1.max()), colorMap=pg.colormap.get('viridis'))
        color_bar1.setImageItem(image_item1, insert_in=plot1)

        # Second plot (interactive with horizontal color bar)
        graphics_widget.nextRow()
        plot2 = graphics_widget.addPlot()
        img2 = np.random.normal(size=(100, 100))
        image_item2 = ImageItem(img2)
        plot2.addItem(image_item2)
        color_bar2 = ColorBarItem(orientation='horizontal', values=(img2.min(), img2.max()), colorMap=pg.colormap.get('viridis'))
        color_bar2.setImageItem(image_item2, insert_in=plot2)

        # Third and fourth plots sharing a color bar
        graphics_widget.nextRow()
        plot3 = graphics_widget.addPlot()
        img3 = np.random.normal(size=(100, 100))
        image_item3 = ImageItem(img3)
        plot3.addItem(image_item3)

        graphics_widget.nextColumn()
        plot4 = graphics_widget.addPlot()
        img4 = np.random.normal(size=(100, 100))
        image_item4 = ImageItem(img4)
        plot4.addItem(image_item4)

        shared_color_bar = ColorBarItem(values=(min(img3.min(), img4.min()), max(img3.max(), img4.max())), colorMap=pg.colormap.get('viridis'))
        shared_color_bar.setImageItem(image_item3, insert_in=plot3)
        shared_color_bar.setImageItem(image_item4, insert_in=plot4)

if __name__ == '__main__':
    app = QtWidgets.QApplication([])
    main_window = MainWindow()
    main_window.show()
    app.exec_()
```