```python
import os
import sys
import argparse
from sirf.STIR import AcquisitionData, AcquisitionModel, ImageData, read_from_file, AcquisitionSensitivityModel

def check_file_exists(file_path, default_path):
    return file_path if os.path.exists(file_path) else default_path

def apply_transform(attenuation_image, transform, transform_type):
    if transform_type == "identity":
        return attenuation_image
    elif transform_type == "custom":
        # Implement custom transform based on the 'transform' parameter
        # This is a placeholder for actual transformation logic
        print(f"Applying custom transform: {transform}")
        return attenuation_image
    else:
        raise ValueError(f"Unsupported transform type: {transform_type}")

def main(args):
    # Check if the provided files exist, otherwise use defaults
    template_sinogram_path = check_file_exists(args.template_sinogram, "default_template.hs")
    attenuation_image_path = check_file_exists(args.attenuation_image, "default_atten_image.hv")
    norm_file_path = check_file_exists(args.norm_file, "default_norm.n.hdr")
    
    # Load the template sinogram
    template_sinogram = read_from_file(template_sinogram_path)
    
    # Create an acquisition model
    acq_model = AcquisitionModel()
    
    # Check if norm and attenuation are present and create an acquisition sensitivity model accordingly
    if args.norm_file and args.attenuation_image:
        norm = AcquisitionData(norm_file_path)
        attenuation_image = read_from_file(attenuation_image_path)
        attenuation_image = apply_transform(attenuation_image, args.transform, args.transform_type)
        
        asm_norm = AcquisitionSensitivityModel(norm)
        asm_atten = AcquisitionSensitivityModel(attenuation_image, acq_model)
        asm = asm_norm * asm_atten
        acq_model.set_acquisition_sensitivity(asm)
    elif args.norm_file:
        norm = AcquisitionData(norm_file_path)
        asm = AcquisitionSensitivityModel(norm)
        acq_model.set_acquisition_sensitivity(asm)
    elif args.attenuation_image:
        attenuation_image = read_from_file(attenuation_image_path)
        attenuation_image = apply_transform(attenuation_image, args.transform, args.transform_type)
        asm = AcquisitionSensitivityModel(attenuation_image, acq_model)
        acq_model.set_acquisition_sensitivity(asm)
    
    # Project the data if normalisation is added
    if args.norm_file:
        # Assuming the projection step here, as the exact requirement is to project data if normalisation is added
        # This could mean different things based on the context (e.g., forward projection of an image)
        # Placeholder for projection logic
    
    # Write the multiplicative sinogram to the specified output file
    # Assuming the output of the acquisition model or sensitivity model is what's meant to be written
    # This is a placeholder for the actual output logic
    print(f"Writing output to {args.output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create multiplicative sinograms from normalisation and/or attenuation data.")
    parser.add_argument("--template_sinogram", help="Path to the template sinogram file")
    parser.add_argument("--attenuation_image", help="Path to the attenuation image file")
    parser.add_argument("--norm_file", help="Path to the ECAT8 bin normalisation file")
    parser.add_argument("--output_file", help="Path to the output file", required=True)
    parser.add_argument("--transform", help="Transform for the attenuation image", default="identity")
    parser.add_argument("--transform_type", help="Type of transform (identity, custom)", default="identity")
    parser.add_argument("--non_interactive", help="Run in non-interactive mode", action="store_true")
    
    args = parser.parse_args()
    
    if args.non_interactive:
        print("Running in non-interactive mode")
    
    main(args)
```