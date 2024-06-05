 ```python
import argparse
import numpy as np
from sirf.IO import read_sirtemplate, ListmodeFile
from sirf.Conversion import ListmodeToSinograms

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("data_path", help="path to data files")
    parser.add_argument("listmode_file", help="listmode file")
    parser.add_argument("output_prefix", help="output file prefix")
    parser.add_argument("raw_data_template", help="raw data template")
    parser.add_argument("time_interval", type=int, help="scanning time interval to convert")
    parser.add_argument("reconstruction_engine", help="reconstruction engine")
    parser.add_argument("acquisition_data_storage_scheme", help="acquisition data storage scheme")
    parser.add_argument("--no-interactive", action="store_true", help="disable interactive mode")

    args = parser.parse_args()

    acquisition_data_storage_scheme = args.acquisition_data_storage_scheme
    raw_data_template = args.raw_data_template
    listmode_file = args.listmode_file
    time_interval = args.time_interval
    reconstruction_engine = args.reconstruction_engine
    output_prefix = args.output_prefix
    data_path = args.data_path

    converter = ListmodeToSinograms()
    converter.setInputFile(ListmodeFile(listmode_file))
    converter.setOutputFile(output_prefix + "_sinograms.h5")
    converter.setRawDataTemplate(raw_data_template)
    converter.setTimeInterval(time_interval)
    converter.setAcquisitionDataStorageScheme(acquisition_data_storage_scheme)
    converter.setReconstructionEngine(reconstruction_engine)
    converter.setInteractive(not args.no_interactive)

    converter.setup()
    sinograms = converter.convert()
    acquisition_data = sinograms.getAcquisitionData()

    array_acquisition_data = np.array(acquisition_data)
    print("Acquisition data dimensions:", array_acquisition_data.shape)

    if not args.no_interactive:
        import matplotlib.pyplot as plt
        plt.imshow(array_acquisition_data[0, :, :], cmap="gray")
        plt.show()

    randoms = converter.estimateRandoms()
    array_randoms = np.array(randoms)

    if not args.no_interactive:
        plt.imshow(array_randoms[0, :, :], cmap="gray")
        plt.show()

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("An error occurred:", str(e))
```