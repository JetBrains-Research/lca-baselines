```python
import sys
import numpy as np
import pyqtgraph as pg
from PyQt5.QtWidgets import QApplication, QMainWindow

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("PyQtGraph Scatter Plot Demo")
        self.setGeometry(100, 100, 800, 600)

        widget = pg.GraphicsLayoutWidget()
        self.setCentralWidget(widget)

        # Generate random data
        np.random.seed(42)
        x = np.random.normal(size=1000)
        y = np.random.normal(size=1000)
        size = np.random.normal(10, 5, size=1000)
        color = np.random.normal(20, 10, size=(1000, 4))

        # Plot 1: All spots identical and transform-invariant
        p1 = widget.addPlot(title="Identical and Transform-Invariant")
        spots1 = [{'pos': pos, 'size': 10, 'pen': {'color': 'r', 'width': 2}} for pos in zip(x, y)]
        scatter1 = pg.ScatterPlotItem(spots=spots1)
        scatter1.sigClicked.connect(lambda: self.onClicked(scatter1))
        p1.addItem(scatter1)

        widget.nextRow()

        # Plot 2: Transform-invariant, but not identical
        p2 = widget.addPlot(title="Transform-Invariant, Not Identical")
        spots2 = [{'pos': pos, 'size': size[i], 'pen': {'color': 'b', 'width': 2}} for i, pos in enumerate(zip(x, y))]
        scatter2 = pg.ScatterPlotItem(spots=spots2)
        scatter2.sigClicked.connect(lambda: self.onClicked(scatter2))
        p2.addItem(scatter2)

        widget.nextRow()

        # Plot 3: Not transform-invariant and not identical
        p3 = widget.addPlot(title="Not Transform-Invariant, Not Identical")
        spots3 = [{'pos': pos, 'size': size[i], 'pen': {'color': color[i], 'width': 2}} for i, pos in enumerate(zip(x, y))]
        scatter3 = pg.ScatterPlotItem(spots=spots3)
        scatter3.sigClicked.connect(lambda: self.onClicked(scatter3))
        p3.addItem(scatter3)

        widget.nextRow()

        # Plot 4: Performance test with large scatter plots
        p4 = widget.addPlot(title="Performance Test")
        x_large = np.random.normal(size=10000)
        y_large = np.random.normal(size=10000)
        spots4 = [{'pos': pos, 'size': 5, 'pen': {'color': 'g', 'width': 1}} for pos in zip(x_large, y_large)]
        scatter4 = pg.ScatterPlotItem(spots=spots4)
        scatter4.sigClicked.connect(lambda: self.onClicked(scatter4))
        p4.addItem(scatter4)

    def onClicked(self, scatter):
        scatter.clearSelection()
        scatter.selectPoints()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    main = MainWindow()
    main.show()
    sys.exit(app.exec_())
```