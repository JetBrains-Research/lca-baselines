```python
from paz.abstract import Processor
from paz.backend.image import load_image, show_image
from paz.backend.image import resize_image, draw_boxes
from paz.backend.image import preprocess_image, augment_image
from paz.backend.image import augment_boxes, preprocess_boxes
from paz.abstract import SequentialProcessor
from paz.datasets import VOCDetection
from paz.processors import AugmentDetection
from paz.processors import ShowBoxes

class PreprocessBoxes(Processor):
    def __init__(self):
        super(PreprocessBoxes, self).__init__()

    def forward(self, data):
        return preprocess_boxes(data)

class PreprocessImage(Processor):
    def __init__(self):
        super(PreprocessImage, self).__init__()

    def forward(self, data):
        return preprocess_image(data)

class AugmentImage(Processor):
    def __init__(self):
        super(AugmentImage, self).__init__()

    def forward(self, data):
        return augment_image(data)

class AugmentBoxes(Processor):
    def __init__(self):
        super(AugmentBoxes, self).__init__()

    def forward(self, data):
        return augment_boxes(data)

class DrawBoxData2D(Processor):
    def __init__(self):
        super(DrawBoxData2D, self).__init__()

    def forward(self, data):
        image, boxes = data
        return draw_boxes(image, boxes)

class ShowBoxes(Processor):
    def __init__(self):
        super(ShowBoxes, self).__init__()

    def forward(self, data):
        return show_image(data)

class AugmentDetection(SequentialProcessor):
    def __init__(self):
        super(AugmentDetection, self).__init__()
        self.add(PreprocessImage())
        self.add(AugmentImage())
        self.add(PreprocessBoxes())
        self.add(AugmentBoxes())

def main():
    import tensorflow as tf
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)

    dataset = VOCDetection()
    model = None

    augment_detection = AugmentDetection()
    show_boxes = ShowBoxes()

    for sample in dataset:
        augmented_sample = augment_detection(sample)
        show_boxes(augmented_sample)

if __name__ == "__main__":
    main()
```