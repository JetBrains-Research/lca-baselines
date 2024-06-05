  import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui
import numpy as np

def main():
    app = QtGui.QApplication([])

    win = pg.GraphicsLayoutWidget()
    win.show()

    # Create random data for the plots
    n = 1000
    x = np.random.normal(size=n)
    y = np.random.normal(size=n)

    # Create the plots
    p1 = win.addPlot(title="Identical and transform-invariant")
    p2 = win.addPlot(title="Transform-invariant, but not identical")
    p3 = win.addPlot(title="Not transform-invariant and not identical")
    p4 = win.addPlot(title="Large scatterplot")

    # Add the data to the plots
    p1.plot(x, y, symbol='o', symbolSize=10, symbolPen=None, symbolBrush=(255, 0, 0))
    p2.plot(x, y, symbol='o', symbolSize=10, symbolPen=None, symbolBrush=(0, 255, 0))
    p3.plot(x, y, symbol='o', symbolSize=10, symbolPen=None, symbolBrush=(0, 0, 255))
    p4.plot(x, y, symbol='o', symbolSize=10, symbolPen=None, symbolBrush=(255, 255, 0))

    # Make the plots clickable
    p1.setMouseEnabled(x=True, y=True)
    p2.setMouseEnabled(x=True, y=True)
    p3.setMouseEnabled(x=True, y=True)
    p4.setMouseEnabled(x=True, y=True)

    # Highlight the clicked points
    def clicked(plot, points):
        for point in points:
            plot.plot(point.x(), point.y(), symbol='o', symbolSize=15, symbolPen=None, symbolBrush=(255, 0, 0))

    p1.sigClicked.connect(clicked)
    p2.sigClicked.connect(clicked)
    p3.sigClicked.connect(clicked)
    p4.sigClicked.connect(clicked)

    if __name__ == '__main__':
        pg.exec()

if __name__ == '__main__':
    main()