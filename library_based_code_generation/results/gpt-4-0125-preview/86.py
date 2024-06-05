```python
import sys
import numpy as np
import pyqtgraph as pg
from PyQt5.QtWidgets import QApplication

class ScatterPlotWindow(pg.GraphicsLayoutWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setWindowTitle('PyQtGraph Scatter Plot Demo')

        # Plot 1: All spots identical and transform-invariant
        p1 = self.addPlot(title="Identical and Transform-Invariant")
        spots1 = [{'pos': np.random.normal(size=2), 'data': 1} for _ in range(100)]
        scatter1 = pg.ScatterPlotItem(size=10, pen=pg.mkPen(None), brush=pg.mkBrush(255, 100, 100, 120))
        scatter1.addPoints(spots1)
        scatter1.sigClicked.connect(self.clicked)
        p1.addItem(scatter1)

        self.nextRow()

        # Plot 2: Spots are transform-invariant, but not identical
        p2 = self.addPlot(title="Transform-Invariant, Not Identical")
        spots2 = [{'pos': np.random.normal(size=2), 'size': np.random.randint(5, 10),
                   'pen': {'color': 'r', 'width': 2}, 'brush':pg.mkBrush(100, 255, 100, 120)} for _ in range(100)]
        scatter2 = pg.ScatterPlotItem()
        scatter2.addPoints(spots2)
        scatter2.sigClicked.connect(self.clicked)
        p2.addItem(scatter2)

        self.nextRow()

        # Plot 3: Spots are not transform-invariant and not identical
        p3 = self.addPlot(title="Not Transform-Invariant, Not Identical")
        spots3 = [{'pos': np.random.normal(size=2), 'size': np.random.randint(5, 15),
                   'pen': {'color': 'b', 'width': 2}, 'brush':pg.mkBrush(100, 100, 255, 120)} for _ in range(100)]
        scatter3 = pg.ScatterPlotItem()
        scatter3.addPoints(spots3)
        scatter3.sigClicked.connect(self.clicked)
        p3.addItem(scatter3)

        self.nextRow()

        # Plot 4: Performance test with large scatter plots
        p4 = self.addPlot(title="Performance Test")
        spots4 = [{'pos': np.random.normal(size=2), 'size': 5,
                   'pen': {'color': 'g', 'width': 1}, 'brush':pg.mkBrush(255, 255, 100, 120)} for _ in range(10000)]
        scatter4 = pg.ScatterPlotItem()
        scatter4.addPoints(spots4)
        scatter4.sigClicked.connect(self.clicked)
        p4.addItem(scatter4)

    def clicked(self, plot, points):
        for point in points:
            point.setBrush('w')

if __name__ == '__main__':
    app = QApplication(sys.argv)
    main = ScatterPlotWindow()
    main.show()
    sys.exit(app.exec_())
```