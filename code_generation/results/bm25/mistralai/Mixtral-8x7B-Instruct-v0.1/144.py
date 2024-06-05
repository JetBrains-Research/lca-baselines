 import sys
import csv
import re
from pysilfont.font import Font
from pysilfont.ufo import UfeatureFile, UtextFile
from pysilfont.fontinfo import FontInfo
from pysilfont.builder import FTMLBuilder
from pysilfont.output import write_to_file, makeFileName, writeFile, setFileForOutput
from pysilfont.input import readGlyphData
from pysilfont.util import _copyFontInfo, cache_font, loadFont, _lineTo, csvWarning, org_sil_check_family_win_ascent_and_descent

def create_ftml(ufo, csv_file, output_file, font_code, log_file, language_tags, rtl, rendering_check, test_name, font_source, text_scaling, anchor_regex, total_width, xsl_stylesheet):
    font = Font()
    font.fromUFO(ufo)

    font_info = FontInfo()
    font_info.fromUFO(ufo)
    _copyFontInfo(font, font_info)

    builder = FTMLBuilder()

    if language_tags:
        for lang in language_tags:
            builder.addLanguage(lang)

    if rtl:
        builder.setRightToLeft()

    if not rendering_check:
        builder.disableRenderingCheck()

    builder.setTestName(test_name)
    builder.setFontSource(font_source)
    builder.setTextScaling(text_scaling)

    if anchor_regex:
        builder.setAnchorPoints(anchor_regex)

    if total_width:
        builder.setTotalWidth(total_width)

    if xsl_stylesheet:
        builder.setXSLStylesheet(xsl_stylesheet)

    with open(csv_file, newline='') as csvfile:
        glyph_data = list(csv.reader(csvfile))
        glyph_data = glyph_data[1:]  # Skip header row

        for row in glyph_data:
            glyph_name, unencoded_specials, ligatures, lam_alef, diacritic_attachment = row

            if glyph_name:
                glyph = font[glyph_name]

                if unencoded_specials:
                    builder.addUnencodedSpecials(glyph, unencoded_specials)

                if ligatures:
                    builder.addLigatures(glyph, ligatures)

                if lam_alef:
                    builder.addLamAlef(glyph, lam_alef)

                if diacritic_attachment:
                    builder.addDiacriticAttachment(glyph, diacritic_attachment)

                builder.addEncodedCharacter(glyph, glyph_name)

    document = builder.getFTML()

    if log_file:
        with open(log_file, 'w') as log:
            log.write(document.serialize())
    else:
        setFileForOutput(output_file)
        write_to_file(document.serialize())

if __name__ == "__main__":
    ufo = sys.argv[1]
    output_file = makeFileName(sys.argv[2], 'ftml')
    csv_file = sys.argv[3]
    font_code = sys.argv[4]
    log_file = sys.argv[5] if len(sys.argv) > 5 else None
    language_tags = sys.argv[6:6 + len(sys.argv[6:])] if len(sys.argv) > 6 else []
    rtl = bool(int(sys.argv[7])) if len(sys.argv) > 7 else False
    rendering_check = bool(int(sys.argv[8])) if len(sys.argv) > 8 else True
    test_name = sys.argv[9]
    font_source = sys.argv[10]
    text_scaling = float(sys.argv[11]) if len(sys.argv) > 11 else 1.0
    anchor_regex = sys.argv[12] if len(sys.argv) > 12 else None
    total_width = int(sys.argv[13]) if len(sys.argv) > 13 else None
    xsl_stylesheet = sys.argv[14] if len(sys.argv) > 14 else None

    create_ftml(ufo, csv_file, output_file, font_code, log_file, language_tags, rtl, rendering_check, test_name, font_source, text_scaling, anchor_regex, total_width, xsl_stylesheet)