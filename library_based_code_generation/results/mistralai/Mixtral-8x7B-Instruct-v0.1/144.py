 import sys
import argparse
import csv
import re
from pysilfont.misc import open_any
from pysilfont.objects import Document, Glyph, Feature, GlyphData, LangSys, Lookup, Rule, MarkClass, Mark, MarkAttachment
from pysilfont.ufo import UfoFont


def parse_args():
    parser = argparse.ArgumentParser(description="Create FTML document from UFO and CSV")

    parser.add_argument("ufo", help="Input UFO file")
    parser.add_argument("ftml", help="Output FTML file")
    parser.add_argument("csv", help="Glyph info CSV file")
    parser.add_argument("--font-code", default="", help="Font code")
    parser.add_argument("--log", default="", help="Log file name")
    parser.add_argument("--langs", nargs="+", default=["en"], help="List of BCP47 language tags")
    parser.add_argument("--rtl", action="store_true", help="Enable right-to-left feature")
    parser.add_argument("--no-render", action="store_true", help="Disable rendering check")
    parser.add_argument("--test-name", default="", help="Test name")
    parser.add_argument("--font-source", default="", help="Font source")
    parser.add_argument("--text-scale", type=float, default=1.0, help="Text scaling factor")
    parser.add_argument("--anchor-regex", default="", help="Anchor points regular expression")
    parser.add_argument("--total-width", type=int, default=0, help="Total width of all string column")
    parser.add_argument("--xsl", default="", help="XSL stylesheet")

    return parser.parse_args()


def main(args):
    ufo = UfoFont(args.ufo)
    doc = Document()

    with open_any(args.csv) as f:
        reader = csv.DictReader(f)
        for row in reader:
            g = Glyph()
            g.name = row["name"]
            g.unicode = int(row["unicode"], 16)
            if "comment" in row:
                g.comment = row["comment"]
            if "width" in row:
                g.width = int(row["width"])
            if "ligature" in row:
                g.ligature = row["ligature"]
            if "lam-alef" in row:
                g.lam_alef = row["lam-alef"]
            if "diacritic" in row:
                g.diacritic = row["diacritic"]
            if "anchor" in row:
                anchors = re.findall(r"(\w+)=(\d+|\()", row["anchor"])
                for anchor in anchors:
                    g.anchors[anchor[0]] = (int(anchor[1]),)
            doc.glyphs[g.name] = g

    if args.font_code:
        doc.font_code = args.font_code
    if args.log:
        doc.log = args.log
    if args.test_name:
        doc.test_name = args.test_name
    if args.font_source:
        doc.font_source = args.font_source
    if args.text_scale != 1.0:
        doc.text_scale = args.text_scale

    lang_sys_list = []
    for lang in args.langs:
        ls = LangSys("dflt")
        ls.req_feature_index = Feature("GSUB").index
        ls.feature_params = {"Ligature": 1}
        if args.rtl and lang in ["ar", "fa", "he"]:
            ls.req_feature_index = Feature("GSUB_RTL").index
            ls.feature_params = {"Ligature_RTL": 1}
        lang_sys_list.append(ls)
    doc.lang_sys_list = lang_sys_list

    if not args.no_render:
        render_feature = Feature("GDEF")
        doc.features.append(render_feature)
        render_lookup = Lookup("gdef_glyph_positioning")
        render_lookup.mark_class_defs = [
            MarkClass("glyph", "Glyph"),
            MarkClass("mark", "Mark"),
        ]
        render_lookup.marks = [
            Mark("glyph", "GPOS_X", 0, 0),
            Mark("glyph", "GPOS_Y", 0, 1),
            Mark("mark", "GPOS_X", 0, 0),
            Mark("mark", "GPOS_Y", 0, 1),
        ]
        render_lookup.rules = [
            Rule(
                "any",
                [
                    ("glyph", 0, "GPOS_X"),
                    ("glyph", 1, "GPOS_Y"),
                    ("mark", 0, "GPOS_X"),
                    ("mark", 1, "GPOS_Y"),
                ],
                [
                    ("glyph", 0, "GPOS_X"),
                    ("glyph", 1, "GPOS_Y"),
                    ("mark", 0, "GPOS_X"),
                    ("mark", 1, "GPOS_Y"),
                ],
            )
        ]
        render_feature.lookups.append(render_lookup)

    if args.total_width:
        string_feature = Feature("SVG")
        doc.features.append(string_feature)
        string_lookup = Lookup("svg_string")
        string_lookup.mark_class_defs = [
            MarkClass("glyph", "Glyph"),
        ]
        string_lookup.marks = [
            Mark("glyph", "SVG_X", 0, 0),
            Mark("glyph", "SVG_Y", 0, 1),
        ]
        string_lookup.rules = [
            Rule(
                "any",
                [
                    ("glyph", 0, "SVG_X"),
                    ("glyph", 1, "SVG_Y"),
                ],
                [
                    ("glyph", 0, "SVG_X"),
                    ("glyph", 1, "SVG_Y"),
                ],
                mark_attachments=[
                    MarkAttachment(
                        "glyph",
                        "SVG_X",
                        0,
                        args.total_width,
                    ),
                ],
            )
        ]
        string_feature.lookups.append(string_lookup)

    doc.write(args.ftml, xsl=args.xsl)


if __name__ == "__main__":
    main(parse_args())