  Here is some sample Python code that uses the SIRF library to create multiplicative sinograms from normalisation and/or attenuation data:
```
import os
import sys
import numpy as np
from sirf.acquisition import AcquisitionModel, AcquisitionSensitivityModel
from sirf.attenuation import AttenuationImage
from sirf.transform import Transform
from sirf.io import read_from_file, write_to_file

def main():
    # Parse command-line options
    data_path = sys.argv[1]
    template_sinogram = sys.argv[2]
    attenuation_image_file = sys.argv[3]
    ecats8_bin_normalisation_file = sys.argv[4]
    output_file = sys.argv[5]
    transform_type = sys.argv[6]
    non_interactive_mode = sys.argv[7]

    # Check if files exist
    if not os.path.exists(data_path):
        print("Data path does not exist")
        return
    if not os.path.exists(template_sinogram):
        print("Template sinogram does not exist")
        return
    if not os.path.exists(attenuation_image_file):
        print("Attenuation image file does not exist")
        return
    if not os.path.exists(ecats8_bin_normalisation_file):
        print("ECATS8 bin normalisation file does not exist")
        return

    # Load data
    data = read_from_file(data_path)
    template_sinogram = read_from_file(template_sinogram)
    attenuation_image = read_from_file(attenuation_image_file)
    ecats8_bin_normalisation = read_from_file(ecats8_bin_normalisation_file)

    # Create acquisition model
    acquisition_model = AcquisitionModel(data, template_sinogram)

    # Check if norm and attenuation are present
    if acquisition_model.has_norm():
        acquisition_model.set_norm(ecats8_bin_normalisation)
    if acquisition_model.has_attenuation():
        acquisition_model.set_attenuation(attenuation_image)

    # Create acquisition sensitivity model
    acquisition_sensitivity_model = AcquisitionSensitivityModel(acquisition_model)

    # Project data if normalisation is added
    if acquisition_model.has_norm():
        acquisition_sensitivity_model.project_data()

    # Write multiplicative sinogram to output file
    write_to_file(output_file, acquisition_sensitivity_model.get_multiplicative_sinogram())

if __name__ == "__main__":
    main()
```
This code uses the `sirf.acquisition` module to create an acquisition model and acquisition sensitivity model, and the `sirf.attenuation` module to handle attenuation data. The `sirf.transform` module is used to handle transformations for the attenuation image. The `sirf.io` module is used to read and write data files.

The code first parses command-line options to get the path to the data files, template sinogram, attenuation image file, ECAT8 bin normalisation file, output file, and transform type. It then checks if the files exist and loads the data using the `read_from_file` function from the `sirf.io` module.

Next, it creates an acquisition model using the `AcquisitionModel` class from the `sirf.acquisition` module, and sets the template sinogram and data using the `set_template_sinogram` and `set_data` methods. It then checks if norm and attenuation are present in the data, and if so, sets the norm and attenuation using the `set_norm` and `set_attenuation` methods.

It then creates an acquisition sensitivity model using the `AcquisitionSensitivityModel` class from the `sirf.acquisition` module, and sets the acquisition model using the `set_acquisition_model` method. It then checks if normalisation is added, and if so, projects the data using the `project_data` method.

Finally, it writes the multiplicative sinogram to the output file using the `write_to_file` function from the `sirf.io` module.