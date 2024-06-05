```python
import numpy as np
import pyqtgraph as pg
from PyQt5 import QtWidgets
from pyqtgraph import ImageItem, ColorBarItem, GraphicsLayoutWidget

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, *args, **kwargs):
        super(MainWindow, self).__init__(*args, **kwargs)
        
        # Create the main widget for the plots
        self.central_widget = GraphicsLayoutWidget()
        self.setCentralWidget(self.central_widget)
        
        # First plot: non-interactive with vertical color bar
        self.plot1 = self.central_widget.addPlot(row=0, col=0)
        self.plot1.disableAutoRange()
        self.plot1.hideButtons()
        image1 = np.random.normal(size=(100, 100))
        img_item1 = ImageItem(image=image1)
        self.plot1.addItem(img_item1)
        color_bar1 = ColorBarItem(values=(image1.min(), image1.max()), colorMap='viridis')
        color_bar1.setImageItem(img_item1, insert_in=self.plot1)
        
        # Second plot: interactive with horizontal color bar
        self.plot2 = self.central_widget.addPlot(row=1, col=0)
        image2 = np.random.normal(size=(100, 100))
        img_item2 = ImageItem(image=image2)
        self.plot2.addItem(img_item2)
        color_bar2 = ColorBarItem(orientation='horizontal', values=(image2.min(), image2.max()), colorMap='plasma')
        color_bar2.setImageItem(img_item2, insert_in=self.plot2)
        
        # Third and fourth plots: share a color bar
        self.plot3 = self.central_widget.addPlot(row=2, col=0)
        self.plot4 = self.central_widget.addPlot(row=3, col=0)
        image3 = np.random.normal(size=(100, 100))
        image4 = np.random.normal(size=(100, 100))
        img_item3 = ImageItem(image=image3)
        img_item4 = ImageItem(image=image4)
        self.plot3.addItem(img_item3)
        self.plot4.addItem(img_item4)
        shared_color_bar = ColorBarItem(values=(min(image3.min(), image4.min()), max(image3.max(), image4.max())), colorMap='magma')
        shared_color_bar.setImageItem(img_item3, insert_in=self.plot3)
        shared_color_bar.setImageItem(img_item4, insert_in=self.plot4)

if __name__ == '__main__':
    app = QtWidgets.QApplication([])
    main = MainWindow()
    main.show()
    pg.exec()
```