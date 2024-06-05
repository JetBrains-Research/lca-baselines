```python
import os
import sys
import argparse
from sirf.STIR import AcquisitionData, AcquisitionModelUsingRayTracingMatrix, ImageData, read_from_file, AcquisitionSensitivityModel

def check_file_exists(file_path, default_path):
    return file_path if os.path.exists(file_path) else default_path

def apply_transform(image, transform, transform_type):
    # Placeholder for actual transformation logic
    if transform_type == "type1":
        # Apply transformation type 1
        pass
    elif transform_type == "type2":
        # Apply transformation type 2
        pass
    # Add more transformation types as needed
    return image

def main(data_path, template_sinogram_path, attenuation_image_path, norm_file_path, output_file, transform, transform_type, non_interactive):
    # Check if the provided files exist, otherwise use defaults
    data_path = check_file_exists(data_path, "default_data_path")
    template_sinogram_path = check_file_exists(template_sinogram_path, "default_template_sinogram_path")
    attenuation_image_path = check_file_exists(attenuation_image_path, "default_attenuation_image_path")
    norm_file_path = check_file_exists(norm_file_path, "default_norm_file_path")
    
    # Load the necessary data
    template_sinogram = AcquisitionData(template_sinogram_path)
    if attenuation_image_path:
        attenuation_image = read_from_file(attenuation_image_path)
        attenuation_image = apply_transform(attenuation_image, transform, transform_type)
    if norm_file_path:
        norm = AcquisitionData(norm_file_path)
    
    # Create an acquisition model
    acq_model = AcquisitionModelUsingRayTracingMatrix()
    
    # Check if norm and attenuation are present and create an acquisition sensitivity model accordingly
    if norm_file_path and attenuation_image_path:
        asm_norm = AcquisitionSensitivityModel(norm)
        asm_att = AcquisitionSensitivityModel(attenuation_image)
        asm = asm_norm * asm_att
    elif norm_file_path:
        asm = AcquisitionSensitivityModel(norm)
    elif attenuation_image_path:
        asm = AcquisitionSensitivityModel(attenuation_image)
    else:
        asm = None
    
    if asm:
        acq_model.set_acquisition_sensitivity(asm)
    
    # Project the data if normalisation is added
    if norm_file_path:
        projected_data = acq_model.forward(template_sinogram)
    else:
        projected_data = template_sinogram
    
    # Write the multiplicative sinogram to the specified output file
    projected_data.write(output_file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create multiplicative sinograms from normalisation and/or attenuation data.")
    parser.add_argument("--data_path", type=str, help="Path to data files")
    parser.add_argument("--template_sinogram", type=str, help="Template sinogram path")
    parser.add_argument("--attenuation_image_file", type=str, help="Attenuation image file path")
    parser.add_argument("--norm_file", type=str, help="ECAT8 bin normalisation file path")
    parser.add_argument("--output_file", type=str, help="Output file path")
    parser.add_argument("--transform", type=str, help="Transform for attenuation image")
    parser.add_argument("--transform_type", type=str, help="Transform type")
    parser.add_argument("--non_interactive", action='store_true', help="Run in non-interactive mode")
    
    args = parser.parse_args()
    
    main(args.data_path, args.template_sinogram, args.attenuation_image_file, args.norm_file, args.output_file, args.transform, args.transform_type, args.non_interactive)
```