import sys
from sirf.STIR import *

def main():
    parser = OptionParser()
    parser.add_option("--data_path", dest="data_path", default="", help="Path to data files")
    parser.add_option("--listmode_file", dest="listmode_file", default="", help="Listmode file")
    parser.add_option("--output_prefix", dest="output_prefix", default="", help="Output file prefix")
    parser.add_option("--raw_data_template", dest="raw_data_template", default="", help="Raw data template")
    parser.add_option("--time_interval", dest="time_interval", default=1.0, type="float", help="Scanning time interval to convert")
    parser.add_option("--recon_engine", dest="recon_engine", default="OSEM", help="Reconstruction engine")
    parser.add_option("--storage_scheme", dest="storage_scheme", default="file", help="Acquisition data storage scheme")
    parser.add_option("--disable_interactive", action="store_true", dest="disable_interactive", default=False, help="Disable interactive mode")
    
    (options, args) = parser.parse_args()
    
    data_path = options.data_path
    listmode_file = options.listmode_file
    output_prefix = options.output_prefix
    raw_data_template = options.raw_data_template
    time_interval = options.time_interval
    recon_engine = options.recon_engine
    storage_scheme = options.storage_scheme
    disable_interactive = options.disable_interactive
    
    set_verbosity(0)
    
    acq_data = AcquisitionData(raw_data_template)
    
    lm2sino = ListmodeToSinograms()
    lm2sino.set_input(listmode_file)
    lm2sino.set_output_prefix(output_prefix)
    lm2sino.set_template(acq_data)
    lm2sino.set_time_interval(time_interval)
    lm2sino.set_up()
    lm2sino.process()
    
    sinograms = lm2sino.get_output()
    acq_array = sinograms.as_array()
    print(acq_array.shape)
    
    if not disable_interactive:
        acq_array.show()
    
    randoms = lm2sino.estimate_randoms()
    random_array = randoms.as_array()
    if not disable_interactive:
        random_array.show()

try:
    main()
except error as e:
    print("Error: %s" % e)