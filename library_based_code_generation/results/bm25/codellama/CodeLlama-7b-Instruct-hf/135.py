  import sys
from PyQt5.QtWidgets import QApplication, QMainWindow
from PyQt5.QtCore import QTimer
from pyqtgraph import PlotWidget, PlotWindow, scatterPlot, ScatterPlot, plotSelected, _plotArray, plot, _plot, test_limits_and_resize, test_zoom_ratio_with_limits_out_of_range, test_makeARGB_with_human_readable_code, ScatterPlotItem, ScatterPlotWidget, _plotMetaArray, registerPlot, addPlot

class BasicPlottingExamples(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setWindowTitle("Basic plotting examples")

        # Create a grid layout to arrange the plots
        grid = QGridLayout()

        # Create a plot widget for each plot
        self.plot1 = PlotWidget()
        self.plot2 = PlotWidget()
        self.plot3 = PlotWidget()
        self.plot4 = PlotWidget()
        self.plot5 = PlotWidget()
        self.plot6 = PlotWidget()
        self.plot7 = PlotWidget()
        self.plot8 = PlotWidget()

        # Add the plot widgets to the grid layout
        grid.addWidget(self.plot1, 0, 0)
        grid.addWidget(self.plot2, 0, 1)
        grid.addWidget(self.plot3, 1, 0)
        grid.addWidget(self.plot4, 1, 1)
        grid.addWidget(self.plot5, 2, 0)
        grid.addWidget(self.plot6, 2, 1)
        grid.addWidget(self.plot7, 3, 0)
        grid.addWidget(self.plot8, 3, 1)

        # Set the plot titles
        self.plot1.setTitle("Basic array plotting")
        self.plot2.setTitle("Multiple curves")
        self.plot3.setTitle("Drawing with points")
        self.plot4.setTitle("Parametric plot with grid enabled")
        self.plot5.setTitle("Scatter plot with axis labels and log scale")
        self.plot6.setTitle("Updating plot")
        self.plot7.setTitle("Filled plot with axis disabled")
        self.plot8.setTitle("Region selection and zoom on selected region")

        # Set the plot labels
        self.plot1.setLabel("bottom", "X")
        self.plot1.setLabel("left", "Y")
        self.plot2.setLabel("bottom", "X")
        self.plot2.setLabel("left", "Y")
        self.plot3.setLabel("bottom", "X")
        self.plot3.setLabel("left", "Y")
        self.plot4.setLabel("bottom", "X")
        self.plot4.setLabel("left", "Y")
        self.plot5.setLabel("bottom", "X")
        self.plot5.setLabel("left", "Y")
        self.plot6.setLabel("bottom", "X")
        self.plot6.setLabel("left", "Y")
        self.plot7.setLabel("bottom", "X")
        self.plot7.setLabel("left", "Y")
        self.plot8.setLabel("bottom", "X")
        self.plot8.setLabel("left", "Y")

        # Set the plot ranges
        self.plot1.setXRange(-10, 10)
        self.plot1.setYRange(-10, 10)
        self.plot2.setXRange(-10, 10)
        self.plot2.setYRange(-10, 10)
        self.plot3.setXRange(-10, 10)
        self.plot3.setYRange(-10, 10)
        self.plot4.setXRange(-10, 10)
        self.plot4.setYRange(-10, 10)
        self.plot5.setXRange(-10, 10)
        self.plot5.setYRange(-10, 10)
        self.plot6.setXRange(-10, 10)
        self.plot6.setYRange(-10, 10)
        self.plot7.setXRange(-10, 10)
        self.plot7.setYRange(-10, 10)
        self.plot8.setXRange(-10, 10)
        self.plot8.setYRange(-10, 10)

        # Set the plot scales
        self.plot1.setScale("linear")
        self.plot2.setScale("linear")
        self.plot3.setScale("linear")
        self.plot4.setScale("linear")
        self.plot5.setScale("linear")
        self.plot6.setScale("linear")
        self.plot7.setScale("linear")
        self.plot8.setScale("linear")

        # Set the plot grid
        self.plot1.showGrid(True, True)
        self.plot2.showGrid(True, True)
        self.plot3.showGrid(True, True)
        self.plot4.showGrid(True, True)
        self.plot5.showGrid(True, True)
        self.plot6.showGrid(True, True)
        self.plot7.showGrid(True, True)
        self.plot8.showGrid(True, True)

        # Set the plot axis labels
        self.plot1.setLabel("bottom", "X")
        self.plot1.setLabel("left", "Y")
        self.plot2.setLabel("bottom", "X")
        self.plot2.setLabel("left", "Y")
        self.plot3.setLabel("bottom", "X")
        self.plot3.setLabel("left", "Y")
        self.plot4.setLabel("bottom", "X")
        self.plot4.setLabel("left", "Y")
        self.plot5.setLabel("bottom", "X")
        self.plot5.setLabel("left", "Y")
        self.plot6.setLabel("bottom", "X")
        self.plot6.setLabel("left", "Y")
        self.plot7.setLabel("bottom", "X")
        self.plot7.setLabel("left", "Y")
        self.plot8.setLabel("bottom", "X")
        self.plot8.setLabel("left", "Y")

        # Set the plot axis ranges
        self.plot1.setXRange(-10, 10)
        self.plot1.setYRange(-10, 10)
        self.plot2.setXRange(-10, 10)
        self.plot2.setYRange(-10, 10)
        self.plot3.setXRange(-10, 10)
        self.plot3.setYRange(-10, 10)
        self.plot4.setXRange(-10, 10)
        self.plot4.setYRange(-10, 10)
        self.plot5.setXRange(-10, 10)
        self.plot5.setYRange(-10, 10)
        self.plot6.setXRange(-10, 10)
        self.plot6.setYRange(-10, 10)
        self.plot7.setXRange(-10, 10)
        self.plot7.setYRange(-10, 10)
        self.plot8.setXRange(-10, 10)
        self.plot8.setYRange(-10, 10)

        # Set the plot axis scales
        self.plot1.setScale("linear")
        self.plot2.setScale("linear")
        self.plot3.setScale("linear")
        self.plot4.setScale("linear")
        self.plot5.setScale("linear")
        self.plot6.setScale("linear")
        self.plot7.setScale("linear")
        self.plot8.setScale("linear")

        # Set the plot grid
        self.plot1.showGrid(True, True)
        self.plot2.showGrid(True, True)
        self.plot3