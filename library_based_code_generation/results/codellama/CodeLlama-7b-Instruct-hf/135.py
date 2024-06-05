  import sys
from PyQt5.QtWidgets import QApplication, QMainWindow
from PyQt5.QtCore import QTimer
from pyqtgraph import PlotWidget, plot

class BasicPlottingExamples(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setWindowTitle("Basic plotting examples")
        self.setGeometry(100, 100, 800, 600)

        self.gridLayout = QGridLayout()
        self.setLayout(self.gridLayout)

        self.plot1 = PlotWidget()
        self.plot1.plot(x=[1, 2, 3], y=[1, 2, 3])
        self.gridLayout.addWidget(self.plot1, 0, 0)

        self.plot2 = PlotWidget()
        self.plot2.plot(x=[1, 2, 3], y=[1, 2, 3], pen='r')
        self.gridLayout.addWidget(self.plot2, 0, 1)

        self.plot3 = PlotWidget()
        self.plot3.plot(x=[1, 2, 3], y=[1, 2, 3], pen='b')
        self.gridLayout.addWidget(self.plot3, 1, 0)

        self.plot4 = PlotWidget()
        self.plot4.plot(x=[1, 2, 3], y=[1, 2, 3], pen='g')
        self.gridLayout.addWidget(self.plot4, 1, 1)

        self.plot5 = PlotWidget()
        self.plot5.plot(x=[1, 2, 3], y=[1, 2, 3], pen='y')
        self.gridLayout.addWidget(self.plot5, 2, 0)

        self.plot6 = PlotWidget()
        self.plot6.plot(x=[1, 2, 3], y=[1, 2, 3], pen='m')
        self.gridLayout.addWidget(self.plot6, 2, 1)

        self.plot7 = PlotWidget()
        self.plot7.plot(x=[1, 2, 3], y=[1, 2, 3], pen='c')
        self.gridLayout.addWidget(self.plot7, 3, 0)

        self.plot8 = PlotWidget()
        self.plot8.plot(x=[1, 2, 3], y=[1, 2, 3], pen='k')
        self.gridLayout.addWidget(self.plot8, 3, 1)

        self.plot9 = PlotWidget()
        self.plot9.plot(x=[1, 2, 3], y=[1, 2, 3], pen='w')
        self.gridLayout.addWidget(self.plot9, 4, 0)

        self.plot10 = PlotWidget()
        self.plot10.plot(x=[1, 2, 3], y=[1, 2, 3], pen='b')
        self.gridLayout.addWidget(self.plot10, 4, 1)

        self.plot11 = PlotWidget()
        self.plot11.plot(x=[1, 2, 3], y=[1, 2, 3], pen='r')
        self.gridLayout.addWidget(self.plot11, 5, 0)

        self.plot12 = PlotWidget()
        self.plot12.plot(x=[1, 2, 3], y=[1, 2, 3], pen='g')
        self.gridLayout.addWidget(self.plot12, 5, 1)

        self.plot13 = PlotWidget()
        self.plot13.plot(x=[1, 2, 3], y=[1, 2, 3], pen='b')
        self.gridLayout.addWidget(self.plot13, 6, 0)

        self.plot14 = PlotWidget()
        self.plot14.plot(x=[1, 2, 3], y=[1, 2, 3], pen='g')
        self.gridLayout.addWidget(self.plot14, 6, 1)

        self.plot15 = PlotWidget()
        self.plot15.plot(x=[1, 2, 3], y=[1, 2, 3], pen='b')
        self.gridLayout.addWidget(self.plot15, 7, 0)

        self.plot16 = PlotWidget()
        self.plot16.plot(x=[1, 2, 3], y=[1, 2, 3], pen='g')
        self.gridLayout.addWidget(self.plot16, 7, 1)

        self.plot17 = PlotWidget()
        self.plot17.plot(x=[1, 2, 3], y=[1, 2, 3], pen='b')
        self.gridLayout.addWidget(self.plot17, 8, 0)

        self.plot18 = PlotWidget()
        self.plot18.plot(x=[1, 2, 3], y=[1, 2, 3], pen='g')
        self.gridLayout.addWidget(self.plot18, 8, 1)

        self.plot19 = PlotWidget()
        self.plot19.plot(x=[1, 2, 3], y=[1, 2, 3], pen='b')
        self.gridLayout.addWidget(self.plot19, 9, 0)

        self.plot20 = PlotWidget()
        self.plot20.plot(x=[1, 2, 3], y=[1, 2, 3], pen='g')
        self.gridLayout.addWidget(self.plot20, 9, 1)

        self.plot21 = PlotWidget()
        self.plot21.plot(x=[1, 2, 3], y=[1, 2, 3], pen='b')
        self.gridLayout.addWidget(self.plot21, 10, 0)

        self.plot22 = PlotWidget()
        self.plot22.plot(x=[1, 2, 3], y=[1, 2, 3], pen='g')
        self.gridLayout.addWidget(self.plot22, 10, 1)

        self.plot23 = PlotWidget()
        self.plot23.plot(x=[1, 2, 3], y=[1, 2, 3], pen='b')
        self.gridLayout.addWidget(self.plot23, 11, 0)

        self.plot24 = PlotWidget()
        self.plot24.plot(x=[1, 2, 3], y=[1, 2, 3], pen='g')
        self.gridLayout.addWidget(self.plot24, 11, 1)

        self.plot25 = PlotWidget()
        self.plot25.plot(x=[1, 2, 3], y=[1, 2, 3], pen='b')
        self.gridLayout.addWidget(self.plot25, 12, 0)

        self.plot26 = PlotWidget()
        self.plot26.plot(x=[1, 2, 3], y=[1, 2, 3], pen='g')
        self.gridLayout.addWidget(self.plot26, 12, 1)

        self.plot27 = PlotWidget()
        self.plot27.plot(x=[1, 2, 3], y=[1, 2, 3], pen='b')
        self.gridLayout.addWidget(self.plot27, 13, 0)

        self.plot28 = PlotWidget()
        self.plot28.plot(x=[1, 2, 3], y=[1, 2, 3], pen='g')
        self.gridLayout.addWidget(self.plot28, 13, 1)

        self.plot29 = PlotWidget()
        self.plot2