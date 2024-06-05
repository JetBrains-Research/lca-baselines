import argparse

# Parse command line arguments
parser = argparse.ArgumentParser(description='Parse command line arguments')
parser.add_argument('--detection_model_path', type=str, help='Path to detection model')
parser.add_argument('--recognition_model_path', type=str, help='Path to recognition model')
parser.add_argument('--table_recognition_model_path', type=str, help='Path to table recognition model')
parser.add_argument('--recognition_model_label_file', type=str, help='Path to recognition model label file')
parser.add_argument('--table_recognition_dict_path', type=str, help='Path to table recognition dictionary')
parser.add_argument('--recognition_batch_size', type=int, help='Recognition model inference batch size')
parser.add_argument('--test_image_path', type=str, help='Path to test image file')
parser.add_argument('--inference_device_type', type=str, help='Inference device type')
parser.add_argument('--device_id', type=int, help='Device ID')
parser.add_argument('--inference_backend_type', type=str, help='Inference backend type')
args = parser.parse_args()

# Build runtime options for models
detection_runtime_options = {}
recognition_runtime_options = {}
table_recognition_runtime_options = {}

# Set backend and device for each model
if args.inference_backend_type == 'TensorRT':
    detection_runtime_options['backend'] = 'TensorRT'
    detection_runtime_options['dynamic_shape'] = True
    detection_runtime_options['trt_cache_file'] = 'detection_trt_cache.bin'
    
    recognition_runtime_options['backend'] = 'TensorRT'
    recognition_runtime_options['dynamic_shape'] = True
    recognition_runtime_options['trt_cache_file'] = 'recognition_trt_cache.bin'
    
    table_recognition_runtime_options['backend'] = 'TensorRT'
    table_recognition_runtime_options['dynamic_shape'] = True
    table_recognition_runtime_options['trt_cache_file'] = 'table_recognition_trt_cache.bin'
else:
    detection_runtime_options['backend'] = 'ONNX'
    recognition_runtime_options['backend'] = 'ONNX'
    table_recognition_runtime_options['backend'] = 'ONNX'

detection_runtime_options['device'] = args.inference_device_type
detection_runtime_options['device_id'] = args.device_id

recognition_runtime_options['device'] = args.inference_device_type
recognition_runtime_options['device_id'] = args.device_id

table_recognition_runtime_options['device'] = args.inference_device_type
table_recognition_runtime_options['device_id'] = args.device_id

# Load models and set preprocessor and postprocessor parameters
# Create PPStructureV2Table instance with loaded models and set recognition batch size

# Read input image, predict and print results, visualize results, and save visualized image