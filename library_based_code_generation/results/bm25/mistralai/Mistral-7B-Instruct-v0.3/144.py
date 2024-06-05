 Here is a basic structure of the Python script using the `pysilfont` library to create an FTML document from a UFO and a glyph data CSV file. Please note that this is a simplified version and might need further adjustments based on your specific requirements.

```python
import argparse
import csv
import pysilfont
from pysilfont.ftml import FTMLBuilder

def read_csv(file_path):
    # Read glyph data from CSV file
    pass

def init_ftml(test_name, font_source, font_code, log_file, rtl, render_check):
    # Initialize FTML document
    pass

def add_glyph_data(ftml_builder, glyph_data):
    # Add encoded characters, unencoded specials and ligatures, Lam-Alef data, and diacritic attachment data
    pass

def main(args):
    # Parse command line arguments
    pass

if __name__ == "__main__":
    main(parse_args())

def parse_args():
    parser = argparse.ArgumentParser(description="Generate FTML document")
    parser.add_argument("ufo", help="Input UFO file")
    parser.add_argument("csv", help="Glyph info CSV file")
    parser.add_argument("output", help="Output FTML file")
    parser.add_argument("--font_code", help="Font code")
    parser.add_argument("--log_file", help="Log file name")
    parser.add_argument("--bcp47", nargs='+', help="List of BCP47 language tags")
    parser.add_argument("--rtl", action="store_true", help="Enable right-to-left feature")
    parser.add_argument("--render_check", action="store_false", help="Disable rendering check")
    parser.add_argument("--test_name", help="Test name")
    parser.add_argument("--font_source", help="Font source")
    parser.add_argument("--text_scaling", type=float, help="Text scaling")
    parser.add_argument("--anchor_points_regex", help="Anchor points regular expression")
    parser.add_argument("--total_width", type=int, help="Total width of all string column")
    parser.add_argument("--xsl", help="XSL stylesheet")

    return parser.parse_args()
```

You will need to implement the functions `read_csv`, `init_ftml`, `add_glyph_data`, and `main` according to the given instructions. The `pysilfont` library provides the necessary APIs for initializing the FTML document, reading glyph data, and adding glyph data to the document. You can find more information about the library's APIs in the official documentation.