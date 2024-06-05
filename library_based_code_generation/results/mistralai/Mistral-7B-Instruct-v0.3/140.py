 Here is a Python script that follows your instructions. I've used the argparse library for parsing command line arguments and the PaddleOCR library for the PPOCR tasks. Please note that you need to install the PaddleOCR library before running this script.

```python
import argparse
import paddle
import paddle.vision.bbox_utils as bbox_utils
import paddle.vision.draw_utils as draw_utils
import cv2
import numpy as np
from paddleocr import PaddleOCR, OcrModel, tools

parser = argparse.ArgumentParser(description="PPOCR Inference")

parser.add_argument("--detection_model", type=str, required=True, help="Path to detection model")
parser.add_argument("--recognition_model", type=str, required=True, help="Path to recognition model")
parser.add_argument("--table_recognition_model", type=str, required=True, help="Path to table recognition model")
parser.add_argument("--label_file", type=str, required=True, help="Path to recognition model label file")
parser.add_argument("--table_recognition_dict", type=str, required=True, help="Path to table recognition dictionary")
parser.add_argument("--batch_size", type=int, default=1, help="Recognition model inference batch size")
parser.add_argument("--test_image", type=str, required=True, help="Path to test image file")
parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "gpu"], help="Inference device type")
parser.add_argument("--device_id", type=int, default=0, help="Device ID")
parser.add_argument("--backend", type=str, default="inference", choices=["inference", "trt"], help="Inference backend type")

args = parser.parse_args()

if args.backend == "trt":
    paddle.enable_static()
    exe, model, feed_var_names, fetch_var_names = paddle.jit.to_static(
        PaddleOCR(),
        feed_dict={"img": paddle.static.data(name="img", shape=[1, 3, 608, 1920], dtype="float32")},
        target_device="cuda:{}".format(args.device_id),
        output_dir="./trt_cache"
    )

device = paddle.device.CUDADevice(args.device_id) if args.device == "gpu" else paddle.device.CPU()

ocr_instance = PaddleOCR(use_gpu=args.device == "gpu", use_8bit=False)
ocr_instance.set_detection_model_path(args.detection_model)
ocr_instance.set_recognition_model_path(args.recognition_model)
ocr_instance.set_table_recognition_model_path(args.table_recognition_model)
ocr_instance.set_use_gpu(args.device == "gpu")
ocr_instance.set_use_8bit(False)
ocr_instance.set_label_file(args.label_file)
ocr_instance.set_table_recognition_dict(args.table_recognition_dict)
ocr_instance.set_recognition_model_inference_batch_size(args.batch_size)

table_ocr = tools.TableOCR(ocr_instance)

img = cv2.imread(args.test_image)
result = table_ocr.ocr(img, use_gpu=args.device == "gpu")

for line in result:
    line_text = " ".join([word_info[-1] for word_info in line])
    print(line_text)

vis_img = draw_utils.draw_ocr_result_on_image(img, result)
cv2.imwrite("result.jpg", vis_img)
```

This script parses command line arguments, sets up the PPOCR models, and performs inference on a test image. It also saves the visualized result as a JPEG image. If the TensorRT backend is used, it generates a TRT cache file in the "./trt_cache" directory.