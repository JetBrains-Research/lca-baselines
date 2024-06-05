import sys
from sirf.STIR import *

def main(argv):
    data_path = 'default_data_path'
    template_sino = 'default_template_sino'
    atten_image = 'default_atten_image'
    norm_file = 'default_norm_file'
    output_file = 'default_output_file'
    atten_transform = 'default_atten_transform'
    transform_type = 'default_transform_type'
    non_interactive = False

    # Parse command-line options
    # Code to parse command-line options goes here

    # Check if provided files exist, if not use default files
    # Code to check if files exist goes here

    # Create acquisition model
    am = AcquisitionModel(data_path, template_sino)

    # Check if norm and attenuation are present
    if norm_file != 'default_norm_file' and atten_image != 'default_atten_image':
        # Create acquisition sensitivity model
        asm = AcquisitionSensitivityModel(norm_file, atten_image, atten_transform, transform_type)

        # Project the data
        am.set_acquisition_sensitivity(asm)
        am.forward_project(template_sino)

    # Write multiplicative sinogram to output file
    am.write_multiplicative_sinogram(output_file)

if __name__ == '__main__':
    main(sys.argv[1:])