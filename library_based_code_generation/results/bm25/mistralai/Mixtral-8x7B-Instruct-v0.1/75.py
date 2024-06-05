 import paz
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

class PreprocessBoxes:
def __init__(self, match_iou_threshold, class_name_to_id):
self.match_iou_threshold = match_iou_threshold
self.class_name_to_id = class_name_to_id

def __call__(self, boxes, boxes_classes):
matched_boxes, matched_classes = paz.match_boxes(
boxes, boxes_classes, self.match_iou_threshold, self.class_name_to_id
)
encoded_boxes = paz.encode_boxes(matched_boxes, to_normalized=True)
one_hot_classes = paz.box_class_to_one_hot(matched_classes, self.class_name_to_id)
return encoded_boxes, one_hot_classes

class PreprocessImage:
def __init__(self, image_size, mean_rgb=(123.675, 116.28, 103.53)):
self.image_size = image_size
self.mean_rgb = mean_rgb

def __call__(self, image):
image = paz.resize_image(image, self.image_size)
image = paz.subtract_mean_image(image, self.mean_rgb)
return image

class AugmentImage:
def __init__(self, image_size, mean_rgb=(123.675, 116.28, 103.53)):
self.image_size = image_size
self.mean_rgb = mean_rgb

def __call__(self, image):
image = paz.resize_image(image, self.image_size)
image = paz.random_image_quality(image)
image = paz.random_image_blur(image)
image = paz.random_image_contrast(image)
image = paz.random_image_brightness(image)
image = paz.random_image_saturation(image)
image = paz.random_image_hue(image)
image = paz.subtract_mean_image(image, self.mean_rgb)
return image

class AugmentBoxes:
def __init__(self, image_size):
self.image_size = image_size

def __call__(self, boxes, boxes_classes):
boxes = paz.ToImageBoxCoordinates(boxes, self.image_size)
boxes = paz._compute_bounding_boxes(boxes)
boxes = paz.test_to_image_coordinates_pass_by_value(boxes)
boxes = paz.expand_boxes(boxes, expand_ratio=1.2)
boxes = paz.random_sample_crop_boxes(boxes, self.image_size)
boxes = paz.random_flip_boxes(boxes)
return boxes

class DrawBoxData2D:
def __init__(self, class_name_to_id):
self.class_name_to_id = class_name_to_id

def __call__(self, image, boxes, boxes_classes):
image = paz.draw_boxes(image, boxes, boxes_classes, self.class_name_to_id)
return image

class ShowBoxes:
def __init__(self, class_name_to_id):
self.class_name_to_id = class_name_to_id

def __call__(self, image, boxes, boxes_classes):
image = paz.resize_image(image, (600, 600))
boxes = paz.decode_boxes(boxes, to_original=True)
boxes = paz.denormalize_boxes(boxes, image.shape[:2])
image = paz.draw_boxes(image, boxes, boxes_classes, self.class_name_to_id)
plt.imshow(image)
plt.show()

class AugmentDetection:
def __init__(self, image_processor, box_processor, training):
self.image_processor = image_processor
self.box_processor = box_processor
self.training = training

def __call__(self, image, boxes, boxes_classes):
if self.training:
image = self.image_processor(image)
boxes = self.box_processor(boxes, boxes_classes)
boxes_classes = paz.one_hot_to_box_class(boxes_classes, self.box_processor.class_name_to_id)
return image, boxes, boxes_classes
else:
image = self.image_processor(image)
return image, boxes, boxes_classes

if __name__ == "__main__":
physical_devices = tf.config.list_physical_devices("GPU")
tf.config.experimental.set_memory_growth(physical_devices[0], True)

model = paz.yolov3()
model.load_weights("yolov3.tf")

dataset = paz.COCODetection("path/to/coco/dataset")

processors = [
PreprocessImage(image_size=416),
PreprocessBoxes(match_iou_threshold=0.5, class_name_to_id=dataset.class_name_to_id),
AugmentImage(image_size=416),
AugmentBoxes(image_size=416),
DrawBoxData2D(class_name_to_id=dataset.class_name_to_id),
ShowBoxes(class_name_to_id=dataset.class_name_to_id),
AugmentDetection(
image_processor=PreprocessImage(image_size=416),
box_processor=PreprocessBoxes(match_iou_threshold=0.5, class_name_to_id=dataset.class_name_to_id),
training=True,
),
]

for sample in dataset:
image, boxes, boxes_classes = sample
for processor in processors:
image, boxes, boxes_classes = processor(image, boxes, boxes_classes)