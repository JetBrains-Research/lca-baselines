 ```python
import sys
from silx.gui import qt, exceptions
from silx.gui.widgets import WaitingPushButton, ThreadPoolPushButton, RangeSlider, LegendIconWidget, ElidedLabel

class MyApp(qt.QMainWindow):

    def __init__(self):
        super().__init__()

        self.create_widgets()
        self.create_layout()
        self.create_connections()

    def create_widgets(self):
        self.waiting_button = WaitingPushButton('Waiting Push Button')
        self.thread_pool_button = ThreadPoolPushButton('Compute Power', callback=self.compute_power)
        self.range_slider = RangeSlider(1, 100, 1, 5, orientation=qt.Qt.Horizontal)
        self.legend_icon_widget = LegendIconWidget()
        self.elided_label = ElidedLabel('This is a long text that might be elided.')

    def create_layout(self):
        layout = qt.QVBoxLayout()

        layout.addWidget(self.waiting_button)
        layout.addWidget(self.thread_pool_button)
        layout.addWidget(self.range_slider)
        layout.addWidget(self.legend_icon_widget)
        layout.addWidget(self.elided_label)

        central_widget = qt.QWidget()
        central_widget.setLayout(layout)

        self.setCentralWidget(central_widget)

    def create_connections(self):
        self.range_slider.valueChanged[int, int].connect(self.print_slider_event)
        self.range_slider.positionChanged[int].connect(self.print_slider_event)

        self.legend_icon_widget.currentChanged.connect(self.print_legend_icon_event)

    def compute_power(self, number):
        result = number ** 2
        print(f'Computed power: {result}')

    def print_slider_event(self, value, position):
        print(f'RangeSlider event: value={value}, position={position}')

    def print_legend_icon_event(self, index):
        item = self.legend_icon_widget.currentItem()
        print(f'LegendIconWidget event: index={index}, item={item}')

def main():
    app = qt.QApplication(sys.argv)

    try:
        window = MyApp()
        window.show()
        sys.exit(app.exec_())

    except exceptions.RuntimeError as e:
        print(e)
        sys.exit(1)

if __name__ == '__main__':
    main()
```