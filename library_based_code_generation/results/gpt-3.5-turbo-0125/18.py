```python
import sys
from silx.gui import qt
from silx.gui.widgets.WaitingPushButton import WaitingPushButton
from silx.gui.widgets.ThreadPoolPushButton import ThreadPoolPushButton
from silx.gui.widgets.RangeSlider import RangeSlider
from silx.gui.plot.LegendIconWidget import LegendIconWidget
from silx.gui.widgets.ElidedLabel import ElidedLabel

app = qt.QApplication(sys.argv)

main_window = qt.QMainWindow()
main_window.setWindowTitle('Simple GUI Application')

waiting_push_button = WaitingPushButton('WaitingPushButton')
def on_waiting_push_button_clicked():
    waiting_push_button.setWaiting(not waiting_push_button.isWaiting())
waiting_push_button.clicked.connect(on_waiting_push_button_clicked)

thread_pool_push_button = ThreadPoolPushButton('ThreadPoolPushButton')
def on_thread_pool_push_button_clicked():
    result = qt.QThreadPool.globalInstance().start(lambda: print(2**10))
thread_pool_push_button.clicked.connect(on_thread_pool_push_button_clicked)

range_slider = RangeSlider('RangeSlider')
def on_range_slider_value_changed(value):
    print('Value changed:', value)
def on_range_slider_position_changed(position):
    print('Position changed:', position)
range_slider.sigValueChanged.connect(on_range_slider_value_changed)
range_slider.sigPositionChanged.connect(on_range_slider_position_changed)

legend_icon_widget = LegendIconWidget('LegendIconWidget')

elided_label = ElidedLabel('ElidedLabel')
elided_label.setText('This is a long text that will be elided if it exceeds the label width.')

layout = qt.QVBoxLayout()
layout.addWidget(waiting_push_button)
layout.addWidget(thread_pool_push_button)
layout.addWidget(range_slider)
layout.addWidget(legend_icon_widget)
layout.addWidget(elided_label)

central_widget = qt.QWidget()
central_widget.setLayout(layout)
main_window.setCentralWidget(central_widget)

main_window.show()

sys.exit(app.exec_())
```