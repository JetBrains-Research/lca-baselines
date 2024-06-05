import argparse
import numpy as np
from listmode import ListmodeToSinogramsConverter

def main():
    parser = argparse.ArgumentParser(description='Estimate randoms from listmode file and compare with delayed coincidences')
    parser.add_argument('--data_path', type=str, help='Path to data files')
    parser.add_argument('--listmode_file', type=str, help='Listmode file')
    parser.add_argument('--sinograms_prefix', type=str, help='Sinograms file prefix')
    parser.add_argument('--randoms_file', type=str, help='Randoms file')
    parser.add_argument('--raw_data_template', type=str, help='Raw data template')
    parser.add_argument('--time_interval', type=float, help='Scanning time interval')
    parser.add_argument('--reconstruction_engine', type=str, help='Reconstruction engine')
    parser.add_argument('--storage_scheme', type=str, help='Acquisition data storage scheme')
    parser.add_argument('--non_interactive', action='store_true', help='Non-interactive mode')

    args = parser.parse_args()

    converter = ListmodeToSinogramsConverter(args.listmode_file, args.sinograms_prefix, args.randoms_file, args.raw_data_template)
    converter.set_time_interval(args.time_interval)
    converter.set_store_delayed_coincidences(True)

    converter.process_data()
    sinograms = converter.get_sinograms()
    estimated_randoms = np.zeros_like(sinograms)

    # Estimate randoms from delayeds via Maximum Likelihood estimation
    # Write estimated randoms to a file

    acquisition_data = converter.get_acquisition_data()
    print('Acquisition data dimensions:', acquisition_data.shape)
    print('Total number of delayed coincidences:', np.sum(acquisition_data))
    print('Estimated randoms:', np.sum(estimated_randoms))
    print('Max values:', np.max(acquisition_data), np.max(estimated_randoms))

    if not args.non_interactive:
        # Display a single sinogram
        pass

if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print('An error occurred:', e)