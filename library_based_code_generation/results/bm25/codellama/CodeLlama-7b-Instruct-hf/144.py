  import sys
import os
import csv
from pysilfont.core import Font, FTML, UfeatureFile, UtextFile, FontInfo
from pysilfont.utils import makeFileName, writeFile, setFileForOutput, readGlyphData, csvWarning, FTMLBuilder

def main(argv):
    # Parse command line arguments
    input_ufo = argv[0]
    output_file = argv[1]
    glyph_data_csv = argv[2]
    font_code = argv[3]
    log_file_name = argv[4]
    bcp47_language_tags = argv[5]
    right_to_left_feature_enabling = argv[6]
    rendering_check_disabling = argv[7]
    test_name = argv[8]
    font_source = argv[9]
    text_scaling = argv[10]
    anchor_points_regex = argv[11]
    total_width_of_all_string_column = argv[12]
    xsl_stylesheet = argv[13]

    # Initialize FTML document
    ftml = FTML()

    # Add encoded characters, unencoded specials, and ligatures to the document
    for glyph in readGlyphData(glyph_data_csv):
        ftml.addEncodedCharacter(glyph['codepoint'])
        ftml.addUnencodedSpecial(glyph['codepoint'])
        ftml.addLigature(glyph['codepoint'])

    # Add Lam-Alef data to the document
    ftml.addLamAlefData(font_code, bcp47_language_tags, right_to_left_feature_enabling, rendering_check_disabling, test_name, font_source, text_scaling, anchor_points_regex, total_width_of_all_string_column)

    # Add diacritic attachment data to the document
    ftml.addDiacriticAttachmentData(font_code, bcp47_language_tags, right_to_left_feature_enabling, rendering_check_disabling, test_name, font_source, text_scaling, anchor_points_regex, total_width_of_all_string_column)

    # Write output FTML file
    writeFile(output_file, ftml.write_to_file(xsl_stylesheet))

if __name__ == "__main__":
    main(sys.argv[1:])