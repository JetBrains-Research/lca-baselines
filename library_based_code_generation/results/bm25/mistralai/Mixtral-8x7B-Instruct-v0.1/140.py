 import argparse
import json
import cv2
from pathlib import Path
from ppocr.runtime.core.base_option import BaseOption
from ppocr.runtime.core.base_model import BaseModel
from ppocr.runtime.core.base_postprocessor import BasePostprocessor
from ppocr.runtime.core.base_preprocessor import BasePreprocessor
from ppocr.runtime.core.detectors.insightface_detector import InsightFaceDetector
from ppocr.runtime.core.recognizers.insightface_recognizer import InsightFaceRecognizer
from ppocr.runtime.core.recognizers.insightface_table_recognizer import InsightFaceTableRecognizer
from ppocr.runtime.core.structures.pp_structure_v2_table import PPStructureV2Table

def parse_arguments():
parser = argparse.ArgumentParser()
parser.add\_argument("--detection-model-path", required=True, help="Path to the detection model")
parser.add\_argument("--recognition-model-path", required=True, help="Path to the recognition model")
parser.add\_argument("--table-recognition-model-path", required=True, help="Path to the table recognition model")
parser.add\_argument("--recognition-model-label-file", required=True, help="Path to the recognition model label file")
parser.add\_argument("--table-recognition-dictionary-path", required=True, help="Path to the table recognition dictionary")
parser.add\_argument("--recognition-model-inference-batch-size", type=int, default=1, help="Recognition model inference batch size")
parser.add\_argument("--test-image-path", required=True, help="Path to the test image")
parser.add\_argument("--inference-device-type", choices=["CPU", "GPU"], required=True, help="Inference device type")
parser.add\_argument("--device-id", type=int, default=0, help="Device ID")
parser.add\_argument("--inference-backend-type", choices=["ONNXRuntime", "TensorRT"], required=True, help="Inference backend type")
return parser.parse\_args()

def build\_runtime\_options(args):
detection\_model\_options = BaseOption()
recognition\_model\_options = BaseOption()
table\_recognition\_model\_options = BaseOption()

if args.inference\_backend\_type == "TensorRT":
detection\_model\_options.set\_backend("TensorRT")
detection\_model\_options.set\_device(args.inference\_device\_type, args.device\_id)
detection\_model\_options.set\_tensorrt\_dynamic\_shape()
detection\_model\_options.set\_tensorrt\_cache\_file\_path("trt\_cache\_file.cache")

recognition\_model\_options.set\_backend("TensorRT")
recognition\_model\_options.set\_device(args.inference\_device\_type, args.device\_id)
recognition\_model\_options.set\_tensorrt\_dynamic\_shape()
recognition\_model\_options.set\_tensorrt\_cache\_file\_path("trt\_cache\_file.cache")

table\_recognition\_model\_options.set\_backend("TensorRT")
table\_recognition\_model\_options.set\_device(args.inference\_device\_type, args.device\_id)
table\_recognition\_model\_options.set\_tensorrt\_dynamic\_shape()
table\_recognition\_model\_options.set\_tensorrt\_cache\_file\_path("trt\_cache\_file.cache")

return detection\_model\_options, recognition\_model\_options, table\_recognition\_model\_options

def main(args):
detection\_model\_options, recognition\_model\_options, table\_recognition\_model\_options = build\_runtime\_options(args)

detection\_model = BaseModel.load\_model(args.detection\_model\_path, detection\_model\_options)
recognition\_model = BaseModel.load\_model(args.recognition\_model\_path, recognition\_model\_options)
table\_recognition\_model = BaseModel.load\_model(args.table\_recognition\_model\_path, table\_recognition\_model\_options)

detection\_preprocessor = InsightFaceDetector.create\_preprocessor(detection\_model)
detection\_postprocessor = InsightFaceDetector.create\_postprocessor()

recognition\_preprocessor = InsightFaceRecognitionPreprocessor()
recognition\_postprocessor = InsightFaceRecognitionPostprocessor()

structure\_table = PPStructureV2Table(
recognition\_model,
table\_recognition\_model,
recognition\_preprocessor,
recognition\_postprocessor,
args.recognition\_model\_label\_file,
args.table\_recognition\_dictionary\_path,
args.recognition\_model\_inference\_batch\_size
)

test\_image = cv2.imread(str(args.test\_image\_path))

results = structure\_table.predict(test\_image)

print(json.dumps(face\_recognition\_to\_json(results), indent=4))

visualized\_image = structure\_table.visualize\_results(test\_image, results)

cv2.imwrite("visualized\_image.jpg", visualized\_image)

if **name** == "**main**":
args = parse\_arguments()
main(args)