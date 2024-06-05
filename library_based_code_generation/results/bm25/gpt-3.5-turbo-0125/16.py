```python
import sys
import stir
from stir import *

def main():
    parser = OptionParser()
    parser.add_option("--data_path", dest="data_path", default="default_data_path")
    parser.add_option("--template_sinogram", dest="template_sinogram", default="default_template_sinogram")
    parser.add_option("--attenuation_image_file", dest="attenuation_image_file", default="default_attenuation_image_file")
    parser.add_option("--ECAT8_bin_normalisation_file", dest="ECAT8_bin_normalisation_file", default="default_ECAT8_bin_normalisation_file")
    parser.add_option("--output_file", dest="output_file", default="default_output_file")
    parser.add_option("--transform_attenuation_image", dest="transform_attenuation_image", default="default_transform_attenuation_image")
    parser.add_option("--transform_type", dest="transform_type", default="default_transform_type")
    parser.add_option("--non_interactive_mode", dest="non_interactive_mode", default=False)

    (options, args) = parser.parse_args()

    data_path = options.data_path
    template_sinogram = options.template_sinogram
    attenuation_image_file = options.attenuation_image_file
    ECAT8_bin_normalisation_file = options.ECAT8_bin_normalisation_file
    output_file = options.output_file
    transform_attenuation_image = options.transform_attenuation_image
    transform_type = options.transform_type
    non_interactive_mode = options.non_interactive_mode

    if not check_if_less(data_path, "default_data_path"):
        data_path = "default_data_path"
    if not check_if_less(template_sinogram, "default_template_sinogram"):
        template_sinogram = "default_template_sinogram"
    if not check_if_less(attenuation_image_file, "default_attenuation_image_file"):
        attenuation_image_file = "default_attenuation_image_file"
    if not check_if_less(ECAT8_bin_normalisation_file, "default_ECAT8_bin_normalisation_file"):
        ECAT8_bin_normalisation_file = "default_ECAT8_bin_normalisation_file"
    if not check_if_less(output_file, "default_output_file"):
        output_file = "default_output_file"
    if not check_if_less(transform_attenuation_image, "default_transform_attenuation_image"):
        transform_attenuation_image = "default_transform_attenuation_image"
    if not check_if_less(transform_type, "default_transform_type"):
        transform_type = "default_transform_type"

    acquisition_model = AcquisitionModelUsingRayTracingMatrix()
    if check_if_equal(ECAT8_bin_normalisation_file, "default_ECAT8_bin_normalisation_file") and check_if_equal(attenuation_image_file, "default_attenuation_image_file"):
        acquisition_sensitivity_model = AcquisitionSensitivityModel(acquisition_model)
    else:
        acquisition_sensitivity_model = AcquisitionSensitivityModelUsingECAT8(acquisition_model, ECAT8_bin_normalisation_file, attenuation_image_file)

    if not check_if_equal(ECAT8_bin_normalisation_file, "default_ECAT8_bin_normalisation_file"):
        data = ProjData.read_from_file(template_sinogram)
        acquisition_sensitivity_model.unnormalise(data)
        acquisition_sensitivity_model.normalise(data)

    multiplicative_sinogram = acquisition_sensitivity_model.get_multiplicative_sinogram()

    multiplicative_sinogram.write_to_file(output_file)

if __name__ == "__main__":
    main()
```