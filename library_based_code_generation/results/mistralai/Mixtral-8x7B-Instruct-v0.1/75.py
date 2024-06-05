 import numpy as np
import tensorflow as tf
from PIL import Image, ImageDraw, ImageFont
import paz

tf.config.experimental.set_memory_growth(tf.config.experimental.list_physical_devices('GPU'), True)

class PreprocessBoxes:
    def __init__(self, match_iou_threshold: float = 0.5, class_name_to_id: dict = None):
        self.match_iou_threshold = match_iou_threshold
        self.class_name_to_id = class_name_to_id or {}

    def __call__(self, boxes: np.ndarray, class_names: np.ndarray) -> tuple:
        matched_boxes, matched_classes = paz.match_boxes(boxes, class_names, self.class_name_to_id, self.match_iou_threshold)
        encoded_boxes = paz.encode_boxes(matched_boxes, image_height=None, image_width=None)
        one_hot_classes = paz.to_one_hot(matched_classes, len(self.class_name_to_id))
        return encoded_boxes, one_hot_classes

class PreprocessImage:
    def __init__(self, image_size: tuple, mean: tuple = None, std: tuple = None):
        self.image_size = image_size
        self.mean = mean or (0, 0, 0)
        self.std = std or (1, 1, 1)

    def __call__(self, image: np.ndarray) -> np.ndarray:
        resized_image = paz.resize_image(image, self.image_size)
        preprocessed_image = paz.normalize_image(resized_image, self.mean, self.std)
        return preprocessed_image

class AugmentImage:
    def __init__(self, image_size: tuple, mean: tuple = None, std: tuple = None):
        self.image_size = image_size
        self.mean = mean or (0, 0, 0)
        self.std = std or (1, 1, 1)

    def __call__(self, image: np.ndarray, augmentation_probability: float = 0.5) -> np.ndarray:
        if np.random.rand() < augmentation_probability:
            resized_image = paz.resize_image(image, self.image_size)
            cropped_background = paz.load_random_cropped_background()
            blended_image = paz.blend_images(resized_image, cropped_background)
            contrast, brightness, saturation, hue = paz.random_image_transform()
            augmented_image = paz.apply_image_transform(blended_image, contrast, brightness, saturation, hue)
            augmented_image = paz.normalize_image(augmented_image, self.mean, self.std)
        else:
            augmented_image = self.preprocess_image(image)
        return augmented_image

class AugmentBoxes:
    def __init__(self, image_size: tuple):
        self.image_size = image_size

    def __call__(self, boxes: np.ndarray, augmentation_probability: float = 0.5) -> np.ndarray:
        if np.random.rand() < augmentation_probability:
            boxes = paz.convert_boxes_to_image_coordinates(boxes, self.image_size)
            expanded_boxes = paz.expand_boxes(boxes, 0.1)
            cropped_boxes, cropped_image_size = paz.random_sample_crop_boxes(expanded_boxes, self.image_size)
            flipped_boxes = paz.random_flip_boxes(cropped_boxes, image_height=cropped_image_size[0], image_width=cropped_image_size[1])
        else:
            flipped_boxes = boxes
        return flipped_boxes

class DrawBoxData2D:
    def __init__(self, class_name_to_id: dict, font_path: str = None, font_size: int = 16, font_color: tuple = (255, 255, 255)):
        self.class_name_to_id = class_name_to_id
        self.font_path = font_path or 'arial.ttf'
        self.font_size = font_size
        self.font_color = font_color

    def __call__(self, image: np.ndarray, boxes: np.ndarray, classes: np.ndarray) -> np.ndarray:
        draw = ImageDraw.Draw(Image.fromarray(image))
        for box, class_id in zip(boxes, classes):
            class_name = list(self.class_name_to_id.keys())[list(self.class_name_to_id.values()).index(class_id)]
            draw.rectangle([box[0], box[1], box[2], box[3]], outline=(255, 0, 0), width=3)
            draw.text((box[0], box[1] - 10), f'{class_name} {class_id}', font=ImageFont.truetype(self.font_path, self.font_size), fill=self.font_color)
        return image

class ShowBoxes:
    def __init__(self, image_size: tuple):
        self.image_size = image_size

    def __call__(self, image: np.ndarray, boxes: np.ndarray, classes: np.ndarray) -> np.ndarray:
        decoded_boxes = paz.decode_boxes(boxes, self.image_size)
        denormalized_boxes = paz.denormalize_boxes(decoded_boxes, self.image_size)
        draw_boxes = DrawBoxData2D(class_name_to_id=None)(image, denormalized_boxes, classes)
        return draw_boxes

class AugmentDetection:
    def __init__(self, image_processor: PreprocessImage, box_processor: PreprocessBoxes, augmentation_probability: float = 0.5):
        self.image_processor = image_processor
        self.box_processor = box_processor
        self.augmentation_probability = augmentation_probability

    def __call__(self, image: np.ndarray, boxes: np.ndarray, classes: np.ndarray) -> tuple:
        augmented_image = self.image_processor(image, augmentation_probability=self.augmentation_probability)
        augmented_boxes, augmented_classes = self.box_processor(boxes, class_names=classes)
        return augmented_image, augmented_boxes, augmented_classes

if __name__ == '__main__':
    model = paz.load_model('path/to/model.h5')
    dataset = paz.load_dataset('path/to/dataset.tfrecord')

    for sample in dataset:
        image, boxes, classes = sample
        image, boxes, classes = AugmentDetection(PreprocessImage(image_size=(416, 416)), PreprocessBoxes(class_name_to_id=model.class_names))(image, boxes, classes)
        image = ShowBoxes(image_size=(416, 416))(image, boxes, classes)
        paz.show_image(image)