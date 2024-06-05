import sys
from sirf.STIR import *
import matplotlib.pyplot as plt

def main(args):
    try:
        raw_data_file = args[1]
        randoms_data_file = args[2]
        attenuation_correction_file = args[3]
        path_to_norm_atten_files = args[4]
        norm_file = args[5]
        atten_image_file = args[6]
        output_prefix = args[7]
        non_interactive = args[8]

        data = AcquisitionData(raw_data_file)
        randoms = AcquisitionData(randoms_data_file)
        acf = AcquisitionData(attenuation_correction_file)
        norm = AcquisitionData(norm_file)
        atten_image = ImageData(atten_image_file)

        scatter_est = ScatterEstimator()
        scatter_est.set_input(data)
        scatter_est.set_randoms(randoms)
        scatter_est.set_attenuation(acf)
        scatter_est.set_normalization(norm)
        scatter_est.set_attenuation_image(atten_image)

        scatter_est.set_up()
        scatter_est.process()

        if non_interactive != 'True':
            scatter_estimate = scatter_est.get_output()
            plt.plot(scatter_estimate.as_array())
            plt.show()

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == '__main__':
    main(sys.argv)