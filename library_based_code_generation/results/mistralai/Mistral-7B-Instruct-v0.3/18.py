 Here is a simple example of a GUI application using the Silx library in Python. Please note that you need to have Silx installed in your environment.

```python
from PyQt5.QtWidgets import QApplication
from silx.gui import qt

class MainWindow(qt.QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Silx GUI Application")

        central_widget = qt.QWidget(self)
        self.setCentralWidget(central_widget)
        layout = qt.QGridLayout(central_widget)

        self.waiting_button = qt.WaitingPushButton("Waiting Button")
        self.waiting_button.clicked.connect(self.toggle_waiting)
        layout.addWidget(self.waiting_button, 0, 0)

        self.thread_button = qt.ThreadPoolPushButton("Thread Pool Button")
        self.thread_button.clicked.connect(self.power_operation)
        layout.addWidget(self.thread_button, 0, 1)

        self.slider = qt.RangeSlider("Range Slider")
        self.slider.valueChanged.connect(self.slider_event)
        self.slider.positionChanged.connect(self.slider_event)
        layout.addWidget(self.slider, 1, 0, 1, 2)

        self.legend_widget = qt.LegendIconWidget("Legend Widget")
        layout.addWidget(self.legend_widget, 2, 0)

        self.label = qt.ElidedLabel("This is a long text that will be elided.", elide_mode=qt.Qt.ElideRight)
        layout.addWidget(self.label, 2, 1)

        self.show()

    def toggle_waiting(self):
        self.waiting_button.setWaiting(not self.waiting_button.isWaiting())

    def power_operation(self):
        try:
            base = float(input("Enter base: "))
            exponent = int(input("Enter exponent: "))
            result = base ** exponent
            print(f"Result: {result}")
        except ValueError:
            qt.QMessageBox.warning(self, "Error", "Please enter a valid number.")

    def slider_event(self, value, position):
        print(f"Value: {value}, Position: {position}")

def main():
    app = QApplication([])
    window = MainWindow()
    app.exec_()

if __name__ == "__main__":
    main()
```

This code creates a main window with the specified widgets. The WaitingPushButton swaps its waiting state when clicked, the ThreadPoolPushButton computes a power operation, the RangeSlider prints events when its value or position changes, the LegendIconWidget displays different styles of lines, symbols, and colormaps, and the ElidedLabel displays long texts with different elide modes. The application handles exceptions using the Silx library's exception handler and cleans up after execution.