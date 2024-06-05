```python
from paz.abstract import SequentialProcessor
from paz.backend.image import imread
from paz.backend.image import show_image
from paz.backend.image import draw_boxes
from paz.backend.image import resize_image
from paz.backend.image import preprocess_images
from paz.backend.image import preprocess_boxes
from paz.processors import (
    RandomContrast,
    RandomBrightness,
    RandomSaturation,
    RandomHue,
    ToImageBoxCoordinates,
    ExpandImageBoxes,
    RandomSampleCropBoxes,
    RandomFlipBoxesLeftRight,
    MatchBoxesToDefaultBoxes,
    EncodeBoxes,
    ExpandClassLabelToOneHot,
    ProcessorA,
)

image = imread('image.jpg')
boxes = [[100, 100, 200, 200, 0]]
default_boxes = [[0, 0, 300, 300, 0]]

image_augmentation_pipeline = SequentialProcessor()
image_augmentation_pipeline.add(RandomContrast())
image_augmentation_pipeline.add(RandomBrightness())
image_augmentation_pipeline.add(RandomSaturation())
image_augmentation_pipeline.add(RandomHue())

box_augmentation_pipeline = SequentialProcessor()
box_augmentation_pipeline.add(ToImageBoxCoordinates())
box_augmentation_pipeline.add(ExpandImageBoxes())
box_augmentation_pipeline.add(RandomSampleCropBoxes())
box_augmentation_pipeline.add(RandomFlipBoxesLeftRight())

draw_boxes_pipeline = ProcessorA(draw_boxes)
preprocess_boxes_pipeline = SequentialProcessor()
preprocess_boxes_pipeline.add(MatchBoxesToDefaultBoxes(default_boxes))
preprocess_boxes_pipeline.add(EncodeBoxes())
preprocess_boxes_pipeline.add(ExpandClassLabelToOneHot())

processor = SequentialProcessor()
processor.add(image_augmentation_pipeline)
processor.add(box_augmentation_pipeline)
processor.add(draw_boxes_pipeline)
processor.add(preprocess_boxes_pipeline)

processed_image, processed_boxes = processor(image, boxes)
show_image(processed_image)
```