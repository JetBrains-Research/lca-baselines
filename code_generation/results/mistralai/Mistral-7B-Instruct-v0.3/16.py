 Here is a simplified Python script that uses the SIRF library to create multiplicative sinograms. Please note that this is a basic example and might need adjustments to fit your specific needs. Also, the SIRF library is not a standard library, so you might need to install it first.

```python
import argparse
import os
import sys
from sirflib.acquisition import AcquisitionModel, AcquisitionSensitivityModel
from sirflib.projection import Projector
from sirflib.system import System

def check_file_exists(file_path):
    if not os.path.exists(file_path):
        print(f"File '{file_path}' not found. Using default file.")
        return "default"
    return file_path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="data_path")
    parser.add_argument("--template_sinogram", type=str, default="template_sinogram.sin")
    parser.add_argument("--attenuation_image", type=str, default="attenuation_image.img")
    parser.add_argument("--norm_file", type=str, default="ECAT8_bin_normalisation.dat")
    parser.add_argument("--output_file", type=str, default="output.sin")
    parser.add_argument("--transform", type=str, default="identity")
    parser.add_argument("--transform_type", type=str, default="affine")
    parser.add_argument("--non_interactive", action="store_true")

    args = parser.parse_args()

    data_path = check_file_exists(args.data_path)
    template_sinogram = check_file_exists(args.template_sinogram)
    attenuation_image = check_file_exists(args.attenuation_image)
    norm_file = check_file_exists(args.norm_file)

    system = System()
    acquisition_model = AcquisitionModel(template_sinogram, norm_file)

    if args.attenuation_image:
        attenuation = system.read_image(os.path.join(data_path, attenuation_image))
        if args.transform:
            attenuation = system.apply_transform(attenuation, args.transform, args.transform_type)
        acquisition_sensitivity_model = AcquisitionSensitivityModel(acquisition_model, attenuation)

        if not args.non_interactive:
            system.interactive_display(attenuation)

    if args.attenuation_image:
        data = system.read_image(os.path.join(data_path, "data.img"))
        if not args.non_interactive:
            system.interactive_display(data)
        if acquisition_sensitivity_model:
            projected_data = Projector(acquisition_sensitivity_model).project(data)
        else:
            projected_data = Projector(acquisition_model).project(data)
        system.write_image(os.path.join(data_path, args.output_file), projected_data)

if __name__ == "__main__":
    main()
```

This script uses the SIRF library to create a multiplicative sinogram from normalization and attenuation data. It accepts command-line options for the paths to data files, template sinogram, attenuation image file, ECAT8 bin normalization file, output file, transform for attenuation image, transform type, and an option for non-interactive mode. If the provided files do not exist, it uses default files. It also handles different types of transformations for the attenuation image. The main function creates an acquisition model, checks if norm and attenuation are present, and based on that, creates an acquisition sensitivity model. It then projects the data if normalization is added, and finally writes the multiplicative sinogram to the specified output file.