```python
from paz.backend.image import load_image
from paz.processors import SequentialProcessor
from paz.backend.image import augmentors as aug
from paz.backend.image import show_image
from paz.backend.image import draw_boxes
from paz.backend.image import preprocessors as pre
from paz.datasets import VOC
from paz.abstract import Processor

class ImageAugmentation(Processor):
    def __init__(self):
        super(ImageAugmentation, self).__init__()

    def call(self, image):
        image = aug.random_contrast(image)
        image = aug.random_brightness(image)
        image = aug.random_saturation(image)
        image = aug.random_hue(image)
        return image

class BoxAugmentation(Processor):
    def __init__(self):
        super(BoxAugmentation, self).__init__()

    def call(self, image, boxes):
        boxes = pre.convert_boxes_to_array(boxes)
        boxes = pre.expand_boxes(boxes)
        boxes = pre.random_sample_crop(image, boxes)
        boxes = pre.random_flip_left_right(image, boxes)
        return boxes

class DrawBoxes(Processor):
    def __init__(self):
        super(DrawBoxes, self).__init__()

    def call(self, image, boxes):
        return draw_boxes(image, boxes)

class PreprocessBoxes(Processor):
    def __init__(self):
        super(PreprocessBoxes, self).__init__()

    def call(self, image, boxes):
        default_boxes = VOC.default_boxes
        boxes = pre.match_boxes_to_default_boxes(boxes, default_boxes)
        boxes = pre.encode_boxes(boxes, default_boxes)
        boxes = pre.expand_class_label(boxes, VOC.classes)
        return boxes

image_augmentation = ImageAugmentation()
box_augmentation = BoxAugmentation()
draw_boxes = DrawBoxes()
preprocess_boxes = PreprocessBoxes()

image_augmentation_pipeline = SequentialProcessor([image_augmentation])
box_augmentation_pipeline = SequentialProcessor([box_augmentation])
draw_boxes_pipeline = SequentialProcessor([draw_boxes])
preprocess_boxes_pipeline = SequentialProcessor([preprocess_boxes])

processor = SequentialProcessor([image_augmentation, box_augmentation, draw_boxes, preprocess_boxes])

image_path = 'path_to_image.jpg'
image = load_image(image_path)
boxes = [[100, 100, 200, 200, 0]]
image_augmented = image_augmentation_pipeline(image)
boxes_augmented = box_augmentation_pipeline(image_augmented, boxes)
image_with_boxes = draw_boxes_pipeline(image_augmented, boxes_augmented)
boxes_preprocessed = preprocess_boxes_pipeline(image_augmented, boxes_augmented)

show_image(image_with_boxes)
```