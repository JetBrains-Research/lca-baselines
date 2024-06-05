```python
import argparse
import csv
from silfont.ftml_builder import FTMLBuilder
from silfont.core import execute

def main(args=None):
    parser = argparse.ArgumentParser(description="Generate FTML document from UFO and glyph data CSV")
    parser.add_argument("-i", "--input", required=True, help="Input UFO file")
    parser.add_argument("-o", "--output", required=True, help="Output FTML file")
    parser.add_argument("-g", "--glyphcsv", required=True, help="Glyph info CSV file")
    parser.add_argument("-f", "--fontcode", required=True, help="Font code")
    parser.add_argument("-l", "--logfile", required=True, help="Log file name")
    parser.add_argument("-b", "--bcp47", nargs="*", default=[], help="List of BCP47 language tags")
    parser.add_argument("-r", "--rtl", action="store_true", help="Enable right-to-left feature")
    parser.add_argument("-d", "--disable-render-check", action="store_true", help="Disable rendering check")
    parser.add_argument("-t", "--testname", required=True, help="Test name")
    parser.add_argument("-s", "--source", required=True, help="Font source")
    parser.add_argument("-x", "--scale", type=float, default=1.0, help="Text scaling")
    parser.add_argument("-a", "--anchorregex", default=".*", help="Anchor points regular expression")
    parser.add_argument("-w", "--totalwidth", type=int, required=True, help="Total width of all string column")
    parser.add_argument("-y", "--xsl", help="XSL stylesheet")

    args = parser.parse_args(args)

    # Initialize FTML document
    ftml = FTMLBuilder(logger=args.logfile, fontcode=args.fontcode, testname=args.testname, fontsource=args.source,
                       rendercheck=not args.disable_render_check, rtl=args.rtl, langs=args.bcp47, xsl=args.xsl,
                       width=args.totalwidth, scale=args.scale, anchorregex=args.anchorregex)

    # Read glyph data from CSV
    with open(args.glyphcsv, mode='r', newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            if row.get('Encoded'):
                ftml.add_encoded(row['Glyph Name'], row['USV'])
            elif row.get('Ligature'):
                ftml.add_ligature(row['Glyph Name'], row['Characters'])
            elif row.get('LamAlef'):
                ftml.add_lam_alef(row['Glyph Name'], row['Characters'])
            elif row.get('Diacritic'):
                ftml.add_diacritic(row['Base'], row['Mark'], row['Glyph Name'])

    # Write the FTML file
    ftml.write_to_file(args.output)

if __name__ == "__main__":
    execute(None, main, {"ftml": FTMLBuilder})
```