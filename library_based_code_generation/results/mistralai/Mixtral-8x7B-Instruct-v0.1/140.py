 import argparse
import json
import sys
import torch
import trt_pose.engine as engine
from trt_pose.models import pose_resnet50_fast
from trt_pose.utils import trt_posix_dll, trt_tensorrt_engine
from PPOCR import *

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--detection_model_path", required=True, help="Path to the detection model")
    parser.add_argument("--recognition_model_path", required=True, help="Path to the recognition model")
    parser.add_argument("--table_recognition_model_path", required=True, help="Path to the table recognition model")
    parser.add_argument("--recognition_label_file", required=True, help="Path to the recognition model label file")
    parser.add_argument("--table_recognition_dictionary_path", required=True, help="Path to the table recognition dictionary")
    parser.add_argument("--recognition_batch_size", type=int, default=1, help="Recognition model inference batch size")
    parser.add_argument("--test_image_path", required=True, help="Path to the test image")
    parser.add_argument("--inference_device_type", choices=["cpu", "gpu"], default="gpu", help="Inference device type")
    parser.add_argument("--device_id", type=int, default=0, help="Device ID")
    parser.add_argument("--inference_backend_type", choices=["tensorrt", "trt", "tensorrt_engine", "tensorflow"], default="tensorrt", help="Inference backend type")
    return parser.parse_args()

def build_runtime_options(args):
    if args.inference_backend_type in ["tensorrt", "trt"]:
        runtime_options = {"device": args.inference_device_type, "device_id": args.device_id, "backend": "tensorrt"}
        if args.inference_backend_type == "trt":
            runtime_options["use_static"] = False
            runtime_options["dynamic_shape"] = True
            runtime_options["fp16"] = True
    elif args.inference_backend_type == "tensorrt_engine":
        runtime_options = {"device": args.inference_device_type, "device_id": args.device_id, "backend": "tensorrt_engine"}
    elif args.inference_backend_type == "tensorflow":
        runtime_options = {"device": args.inference_device_type, "device_id": args.device_id, "backend": "tensorflow"}
    else:
        raise ValueError("Invalid inference backend type")
    return runtime_options

def load_models(detection_model_path, recognition_model_path, table_recognition_model_path, runtime_options):
    detection_model = DetectionModel(detection_model_path, runtime_options=runtime_options)
    recognition_model = RecognitionModel(recognition_model_path, label_file=args.recognition_label_file, runtime_options=runtime_options)
    table_recognition_model = TableRecognitionModel(table_recognition_model_path, dictionary_path=args.table_recognition_dictionary_path, runtime_options=runtime_options)
    return detection_model, recognition_model, table_recognition_model

def main(args):
    args = parse_arguments()
    runtime_options = build_runtime_options(args)
    detection_model, recognition_model, table_recognition_model = load_models(args.detection_model_path, args.recognition_model_path, args.table_recognition_model_path, runtime_options)

    pp_structure_v2_table = PPStructureV2Table(detection_model, recognition_model, table_recognition_model)
    pp_structure_v2_table.recognition_batch_size = args.recognition_batch_size

    image = cv2.imread(args.test_image_path)
    results = pp_structure_v2_table.predict(image)
    print(json.dumps(results, indent=4))

    visualized_image = pp_structure_v2_table.visualize(image, results)
    cv2.imwrite("output.jpg", visualized_image)

if __name__ == "__main__":
    main(sys.argv)