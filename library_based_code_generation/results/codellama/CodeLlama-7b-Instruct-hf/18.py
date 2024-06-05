  import silx.gui as silx

class MyApp(silx.app.Application):
    def __init__(self):
        super().__init__()
        self.main_window = silx.gui.window.MainWindow()
        self.main_window.setWindowTitle("My GUI Application")

        # Create widgets
        self.waiting_push_button = silx.gui.widgets.WaitingPushButton(self.main_window)
        self.waiting_push_button.setText("Waiting Push Button")
        self.waiting_push_button.clicked.connect(self.on_waiting_push_button_clicked)

        self.thread_pool_push_button = silx.gui.widgets.ThreadPoolPushButton(self.main_window)
        self.thread_pool_push_button.setText("ThreadPool Push Button")
        self.thread_pool_push_button.clicked.connect(self.on_thread_pool_push_button_clicked)

        self.range_slider = silx.gui.widgets.RangeSlider(self.main_window)
        self.range_slider.setLabels(["Min", "Max"])
        self.range_slider.setRange(0, 100)
        self.range_slider.setSingleStep(1)
        self.range_slider.setPageStep(10)
        self.range_slider.valueChanged.connect(self.on_range_slider_value_changed)

        self.legend_icon_widget = silx.gui.widgets.LegendIconWidget(self.main_window)
        self.legend_icon_widget.setStyles(["lines", "symbols", "colormaps"])
        self.legend_icon_widget.setSizePolicy(silx.gui.widgets.LegendIconWidget.SizePolicy.Fixed)

        self.elided_label = silx.gui.widgets.ElidedLabel(self.main_window)
        self.elided_label.setText("This is a very long text that will be elided")
        self.elided_label.setElideMode(silx.gui.widgets.ElidedLabel.ElideMode.Middle)

        # Layout widgets
        layout = silx.gui.layout.Layout()
        layout.addWidget(self.waiting_push_button)
        layout.addWidget(self.thread_pool_push_button)
        layout.addWidget(self.range_slider)
        layout.addWidget(self.legend_icon_widget)
        layout.addWidget(self.elided_label)
        self.main_window.setLayout(layout)

    def on_waiting_push_button_clicked(self):
        self.waiting_push_button.setWaiting(not self.waiting_push_button.isWaiting())

    def on_thread_pool_push_button_clicked(self):
        self.thread_pool_push_button.setEnabled(False)
        self.thread_pool_push_button.setText("Computing...")
        self.thread_pool_push_button.setWaiting(True)
        self.thread_pool_push_button.setEnabled(True)

    def on_range_slider_value_changed(self, value):
        print(f"Range slider value: {value}")

    def on_legend_icon_widget_clicked(self, style):
        print(f"Legend icon widget style: {style}")

    def on_elided_label_clicked(self):
        print("Elided label clicked")

if __name__ == "__main__":
    app = MyApp()
    app.exec_()