 ```python
import argparse
import numpy as np
import sinograms
from likelihood_estimator import MaximumLikelihoodEstimator

def main():
    parser = argparse.ArgumentParser(description='Estimate randoms from a list mode file.')
    parser.add_argument('data_path', help='Path to data files')
    parser.add_argument('listmode_file', help='Listmode file')
    parser.add_argument('sinograms_prefix', help='Sinograms file prefix')
    parser.add_argument('randoms_file', help='Randoms file')
    parser.add_argument('raw_data_template', help='Raw data template')
    parser.add_argument('time_interval', type=int, help='Scanning time interval')
    parser.add_argument('--reconstruction_engine', default='emrecon', help='Reconstruction engine')
    parser.add_argument('--acquisition_data_storage_scheme', default='coincidence', help='Acquisition data storage scheme')
    parser.add_argument('--non_interactive', action='store_true', help='Non-interactive mode')

    args = parser.parse_args()

    try:
        converter = sinograms.ListmodeToSinogramsConverter(
            input_file=args.listmode_file,
            output_file_prefix=args.sinograms_prefix,
            raw_data_template=args.raw_data_template,
            time_interval=args.time_interval,
            reconstruction_engine=args.reconstruction_engine,
            acquisition_data_storage_scheme=args.acquisition_data_storage_scheme
        )

        sinograms_data = converter.process_data()
        sinograms = sinograms_data['sinograms']
        delayed_coincidences = sinograms_data['delayed_coincidences']

        likelihood_estimator = MaximumLikelihoodEstimator(sinograms, delayed_coincidences)
        estimated_randoms = likelihood_estimator.estimate()

        np.savetxt(args.randoms_file, estimated_randoms, fmt='%f')

        acquisition_data = sinograms_data['acquisition_data']
        print(f'Acquisition data dimensions: {acquisition_data.shape}')
        print(f'Total number of delayed coincidences: {np.sum(delayed_coincidences)}')
        print(f'Total number of estimated randoms: {np.sum(estimated_randoms)}')
        print(f'Max acquisition data value: {np.max(acquisition_data)}')

        if not args.non_interactive:
            import matplotlib.pyplot as plt
            plt.imshow(sinograms[0, :, :], cmap='gray')
            plt.show()

    except Exception as e:
        print(f'Error: {e}')

if __name__ == '__main__':
    main()
```