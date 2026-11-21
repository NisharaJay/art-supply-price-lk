"""
Clean the raw scraped data and build model features.

AI-assisted: written with the help of Claude. See AI_USAGE.md.
"""

import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
OUT_DIR = ROOT / "data" / "processed"

# this project studies art supplies, so craft / stationery items are removed.
CRAFT_NAME_PATTERN = (
    r"silicone?\s*mou?ld|\bmou?lds?\b|resin|epoxy|crepe|foam sheet|eva foam|scrap ?book|srapbook|"
    r"die ?cuts?|cutting die|punch|quilting|rotary|dress making|pattern|curve rule|"
    r"glitter|pom ?pom|feather|button|ribbon|bead|doily|cellophane|stamp|decoupage|cake topper|"
    r"dry flower|tassel|wrapping|greeting card|fun crafts|\bcard \d|jute|magnet|buckle|binder ring|"
    r"album|eyelet|brads?\b|pipe cleaner|popsicle|plastic eyes|wax seal|face ?paint|embossing|"
    r"ink blending|napkin|glue gun|drill|bbq|wooden (?:letter|shape|peg)|keychain|earring|coaster|"
    r"photo frame|sticker|tissue paper|quilling|craft kit|mathematical|cleaner|heat ?gun|dig & discover|"
    r"diamond painting|tape runner|glue applicator|silicone cup|"
    # general stationery / office items (not art supplies)
    r"velcro|glue dots|(?<!masking )\btape\b|(?<!leafing )\bglue\b|b6000|jewell?ery|lcd writing|"
    r"stationa?ry kit|white ?board|ballpoint|ener ?gel|\bgel pen|pen gift box|rust off|"
    # papercraft / sewing tools
    r"dye spray|spray shiny|corner rounder|scoreboard|emboss tool|hole cutter|letter cutter|heat erasable|"
    r"clear block|bendy rubber pencil"
)
ART_NAME_PATTERN = (
    r"paint|brush|canvas|sketch|pencil|charcoal|pastel|easel|palett?e|water ?colou?r|acrylic|"
    r"oil colou?r|drawing|artist|gouache|tempera|clay|manikin|gold leaf|fineliner|calligraph"
)
CRAFT_SHOP_CATEGORIES = {
    "silicon molds", "other resin materials", "resin supplies", "resin craft", "epoxy resins, uv resins",
    "rasin", "other crafting materials", "scrapbooking", "dry flowers", "wordings die cuts",
    "die cutting machine", "love", "tags and shapes", "quilling", "clear stamps", "craft", "dress making",
    "baby", "wooden cake toppers", "craft punches", "wooden stamps", "crepe papers", "craft buttons",
    "christmas", "wooden shapes for art & craft", "tissue papers | napkins", "decorative tapes",
    "embossing tools", "craft items", "craft ink pads", "kids crafts", "flower and leaf designs",
    "popsicle sticks and craft sticks", "glitter dust", "pom pom and feathers", "cards & wrapping papers",
    "buckles", "other craft supplies", "handmade greeting cards", "foam sheets",
    "album corners | brads | eyelets", "scoring & folding", "half pearls & stones", "ribbons", "magnets",
    "binder rings", "jute material", "paper doily", "cellophane and envelops", "decoupage", "wax sealing",
    "beads pearls", "jewelry materials", "pipe cleaners", "craft embelishments", "glass message bottle",
    "wooden pegs", "plastic eyes", "card making packs", "face paint", "paper|board|foam|crepe",
    "embossed boards", "stationery items", "haanz gift vouchers", "",
}


def is_craft(name, categories):
    name = name.lower()
    if re.search(CRAFT_NAME_PATTERN, name):
        return True
    cats = {c.strip().lower() for c in str(categories if isinstance(categories, str) else "").split(";")}
    return cats <= CRAFT_SHOP_CATEGORIES and not re.search(ART_NAME_PATTERN, name)


# 1. Category groups: 211 messy shop categories -> 12 clear art groups
CATEGORY_RULES = [
    ("Brushes",                   r"brush"),                                  
    ("Pens & Markers",            r"\bpens?\b|marker(?!\s*pad)|liner|\bink\b|calligraph|posca|micron"),
    ("Pencils, Charcoal & Pastels", r"pencil|charcoal|pastel(?!\s*paint)|crayon|graphite|\bleads?\b"),
    # "medium" but not colour names like "Medium Yellow" / "Red Medium" or sizes like "Small/Medium"
    ("Mediums & Varnishes",       r"(?<!yellow )(?<!red )(?<!/)\bmediums?\b(?! yellow| red| blue| green)|"
                                  r"varnish|gesso|fixative|modell?ing paste|linseed|linceed|turpentine|"
                                  r"drying oil|silicone? oil"),
    ("Fabric & Glass Paint",      r"fabric|glass|ceramic"),
    ("Oil Paint",                 r"\boil\b"),
    ("Watercolour & Gouache",     r"water ?colou?r|gouache"),
    ("Poster & Tempera Paint",    r"poster|tempera|banner"),
    ("Acrylic Paint",             r"acrylic"),
    ("Tools & Accessories",       r"easel|drawing board|palett?e|pallet|knife|knives|cutter|cutting|"
                                  r"eraser|sharpener|ruler|set square|stencil|tool|blender|mixing tray"),
    ("Paper & Canvas",            r"paper|\bpads?\b|board|canvas|sketch ?book|drawing book"),
    ("Clay & Sculpting",          r"clay|sculpt|pottery|porrety|manikin"),
]

DEFAULT_CATEGORY = "Other Art Supplies"   # art kits, gold leaf, aprons


# 2. Brands: the shop "brand" field is empty for 94% of rows, so search the product name + categories
BRAND_RULES = [
    (r"mont ?marte", "Mont Marte"),
    (r"art ?rangers?", "Art Rangers"),
    (r"art ?nation", "Art Nation"),
    (r"marie[’']?s|\bmaries\b", "Marie's"),
    (r"fevicryl", "Fevicryl"),
    (r"faber[ -]?cast(?:el|le)", "Faber-Castell"),   # also "Faber Castle"
    (r"fabriano", "Fabriano"),
    (r"pent[ae]l", "Pentel"),                      # "Pental" is a common misspelling
    (r"sakura|pigma|micron", "Sakura"),
    (r"camlin|camel|fabrica", "Camlin (Camel)"),
    (r"bomeijia", "Bomeijia"),
    (r"sta?edtler|steadtler", "Staedtler"),        # also "Steadtler"
    (r"tombow?\b", "Tombow"),
    (r"posca|\buni\b", "Uni"),
    (r"prisma ?colou?r", "Prismacolor"),          # also "Prisma Colour"
    (r"giorgione", "Giorgione"),
    (r"rotring", "Rotring"),
    (r"nyoni", "Nyoni"),
    (r"fujin", "Fujin"),
    (r"macron", "Macron"),
    (r"winsor|newton", "Winsor & Newton"),
    (r"\breeves\b", "Reeves"),
    (r"derwent", "Derwent"),
    (r"talens|rembrandt|van gogh|art creation", "Royal Talens"),
    (r"superior", "Superior"),
    (r"keep ?smil", "Keep Smiling"),               # also "KeepSmiling", "Keep Smilling"
    (r"\bdoms\b", "DOMS"),
    (r"crimto", "Crimto"),
    (r"\bacron\b", "Acron"),
    (r"worison", "Worison"),
    (r"conqueror", "Conqueror"),
    (r"artline", "Artline"),
    (r"monami", "Monami"),
    (r"\bsapri\b", "Sapri"),
    (r"sea ?king", "SeaKing"),
    (r"pebeo", "Pebeo"),
    (r"mungyo", "Mungyo"),
    (r"bomega", "Bomega"),
    (r"tenwin", "Tenwin"),
    (r"hauser", "Hauser"),
    (r"lakaruchi", "Lakaruchi"),
    (r"\btinto\b", "Tinto"),
    (r"\blantu\b", "Lantu"),
    (r"stabilo", "Stabilo"),
    (r"nataraj", "Nataraj"),
    (r"\bapple\b(?! green)", "Apple (brushes)"),   # not the colour "Apple green"
    (r"\bflash\b", "Flash"),
    (r"hobby ?craft", "Hobby Craft"),
]
MIN_BRAND_COUNT = 10   # brands with fewer products are grouped as "Other brand"


# The product NAME is checked first for these, because shop categories can mislead
# (e.g. "Drawing Papers For Watercolours" is paper, not watercolour paint).
NAME_FIRST_RULES = [
    ("Pens & Markers",        r"(?<!water)(?<!water )brush ?pens?|brush tip|marker(?!\s*pad)"),  # brush pens
    ("Pencils, Charcoal & Pastels", r"colou?r pencils|sketching set"),  # pencil sets that come "+ sharpener"
    ("Tools & Accessories",   r"brush (?:pouch|bucket|holder|storage)|brush pouch|easel|drawing board|"
                              r"tote board|carry board|clip ?board|cutter|trimmer|stump|palett?e|"
                              r"pall?ete?\b|sand ?paper|knife|knives|painting plate|mixing (?:pot|tray|plate)|"
                              r"sharpener|shapner|eraser|\btools?\b"),
    ("Brushes",               r"brush"),
    ("Acrylic Paint",         r"acrylic.*pastel|pastel.*acrylic|art rangers? 75ml"),  # pastel-coloured acrylics
    ("Mediums & Varnishes",   r"fixative|varnish"),
    ("Fabric & Glass Paint",  r"fabric paint|glass colou?rs?|glass paint|dimentional fabric|dye sticks"),
    ("Clay & Sculpting",      r"\bclay\b|manikin"),
    ("Watercolour & Gouache", r"gouache"),
    ("Poster & Tempera Paint", r"poster colou?r"),
    ("Paper & Canvas",        r"\bpapers?\b|\bpads?\b|canvas|sketch ?book|drawing book|art board|dessin|disegno"),
]


def find_category(name, text):
    for group, pattern in NAME_FIRST_RULES:
        if re.search(pattern, name):
            return group
    for group, pattern in CATEGORY_RULES:
        if re.search(pattern, text):
            return group
    return DEFAULT_CATEGORY


def find_brand(brand_field, text):
    search_in = f"{brand_field} {text}".lower()
    for pattern, brand in BRAND_RULES:
        if re.search(pattern, search_in):
            return brand
    if isinstance(brand_field, str) and brand_field.strip():
        return brand_field.strip().title()
    return "Unknown"


def first_number(text, pattern):
    """Return the first number captured by `pattern` (group 1), or NaN."""
    m = re.search(pattern, text)
    return float(m.group(1)) if m else np.nan


def extract_weight_g(text):
    m = re.search(r"(\d+(?:\.\d+)?)\s*(kg|g|gm|gms|grams?)\b", text)
    if not m:
        return np.nan
    value = float(m.group(1))
    return value * 1000 if m.group(2) == "kg" else value


def extract_pack_count(text):
    """How many items in the product, e.g. '12 colours', '5pcs', 'set of 24', '72 set'."""
    patterns = [
        r"(\d+)\s*[x×*]\s*\d+(?:\.\d+)?\s*(?:ml|g)\b",    # "24X12ml", "12x10g" -> 24, 12
        r"\d+(?:\.\d+)?\s*(?:ml|g)\s*[x×*]\s*(\d+)",      # "75ml*6"              -> 6
        r"\d+\s*ml\s*/\s*(\d+)\s*(?:tubes|colou?rs|pcs)",  # "12ml/24 tubes" -> 24 (not "180ml/300ml")
        r"set of (\d+)",
        r"(\d+)\s*-?\s*(?:pcs|pc|pieces|colou?rs?|shades?|sheets|sticks|pens|pencils|brushes|markers|"
        r"tubes|cakes|clrs|crayons|nibs?|leads|pages|pgs)\b",               # "24 Tubes", "-36Clrs", "9 Nibs"
        r"(\d+)\s*/\s*\d+\s*ml",                                              # "12/20ml" = 12 tubes of 20ml
        r"(\d+)\s*(?:set|pack)\b",
        # number at the end of a colour-set name, e.g. "Oil pastel -50", "Colour pencils 24"
        r"(?:pencils?|pastels?|watercolou?rs?|water colou?rs?|crayons|colou?rs?|cakes|tubes|paints?)"
        r"\s*(?:set|pack)?\s*[-–]?\s*(\d{1,3})(?![\d.]|\s*(?:ml|mm|g\b|cm|b\b|h\b|x|pc))",
    ]
    for p in patterns:
        m = re.search(p, text)
        if m and 1 < int(m.group(1)) <= 500:
            return int(m.group(1))
    return 1


def grade(text):
    if re.search(r"\bartists?\b|professional|premier|accademia|academy|master", text):
        return "Professional"
    if re.search(r"student|junior|kids?\b|children|school|\bfun\b", text):
        return "Student/Kids"
    return "Unspecified"


def main():
    raw_files = sorted(RAW_DIR.glob("art_supplies_raw_*.csv"))
    raw_files = [f for f in raw_files if "_TEST" not in f.name]
    raw_path = raw_files[-1]
    df = pd.read_csv(raw_path)
    report = [f"Input: {raw_path.name}", f"Raw rows: {len(df)}", ""]

    #cleaning
    def drop(mask, reason):
        nonlocal df
        report.append(f"Removed {int(mask.sum()):4d} rows: {reason}")
        df = df[~mask].copy()

    drop(df["price_lkr"].isna() | (df["price_lkr"] <= 0), "price missing or Rs. 0")
    drop(df["name"].str.contains("voucher|gift card", case=False, na=False), "gift vouchers (not a product)")
    dup = df.duplicated(subset=["site", "name", "price_lkr"], keep="first")
    drop(dup, "exact duplicates (same shop + same name + same price), e.g. colour variants;"
              " kept once so they don't leak between train and test")
    craft = pd.Series([is_craft(n, c) for n, c in zip(df["name"], df["categories"])], index=df.index)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df[craft][["site", "name", "categories", "price_lkr", "url"]].to_csv(
        OUT_DIR / "excluded_craft_items.csv", index=False, encoding="utf-8-sig")
    drop(craft, "craft / stationery items, not art supplies (list: excluded_craft_items.csv)")
    report.append(f"Rows after cleaning (art supplies only): {len(df)}\n")

    #features
    name = df["name"].str.lower()
    text = (df["name"].fillna("") + " " + df["categories"].fillna("")).str.lower()

    df["category_group"] = [find_category(n, t) for n, t in zip(name, text)]
    df["brand"] = [find_brand(b, t) for b, t in zip(df["brand"].fillna(""), text)]
    counts = df["brand"].value_counts()
    df["brand_grouped"] = df["brand"].where(
        df["brand"].map(counts).ge(MIN_BRAND_COUNT) | df["brand"].eq("Unknown"), "Other brand")

    df["volume_ml"] = name.apply(lambda t: first_number(t, r"(\d+(?:\.\d+)?)\s*ml\b"))
    # litres, e.g. "Topcoat 1L" (but not brush codes like "(113L)")
    litres = name.apply(lambda t: first_number(t, r"(?<![\w(])(\d+(?:\.\d+)?)\s*(?:l|ltr|litres?|liters?)\b"))
    df["volume_ml"] = df["volume_ml"].fillna(litres * 1000)
    df["weight_g"] = name.apply(extract_weight_g)
    df["paper_gsm"] = name.apply(lambda t: first_number(t, r"(\d+)\s*gsm"))
    # on paper/canvas, "380g" means paper weight (gsm), not product weight
    paper_g = (df["category_group"] == "Paper & Canvas") & df["paper_gsm"].isna() & df["weight_g"].notna()
    df.loc[paper_g, "paper_gsm"] = df.loc[paper_g, "weight_g"]
    df.loc[paper_g, "weight_g"] = np.nan
    df["paper_size"] = name.str.extract(r"\b(a[0-6])\b", expand=False).str.upper().fillna("Not stated")
    df["pack_count"] = name.apply(extract_pack_count)
    df["is_set"] = name.str.contains(r"\b(?:set|pack|kit|combo|collection|assorted)\b").astype(int)
    df["grade"] = name.apply(grade) 
    df["is_variable"] = (df["product_type"] == "variable").astype(int)
    df["on_sale"] = df["on_sale"].astype(int)
    df["in_stock"] = df["in_stock"].astype(int)
    df["name_word_count"] = df["name"].str.split().str.len()
    df["log_price"] = np.log1p(df["price_lkr"])

    columns = [
        #identifiers (not model features)
        "site", "product_id", "name", "url",
        #features
        "category_group", "brand", "brand_grouped", "grade", "paper_size",
        "volume_ml", "weight_g", "paper_gsm", "pack_count", "is_set",
        "n_variants", "is_variable", "on_sale", "in_stock", "name_word_count",
        #targets
        "price_lkr", "log_price",
    ]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "art_supplies_clean.csv"
    df[columns].to_csv(out_path, index=False, encoding="utf-8-sig")

    #report
    report += ["Category groups:", df["category_group"].value_counts().to_string(), ""]
    known = (df["brand"] != "Unknown").mean() * 100
    report += [f"Brand identified for {known:.1f}% of products (shop field alone: ~6%)",
               df["brand_grouped"].value_counts().to_string(), ""]
    report += ["Size features found (number of products):"]
    for c in ["volume_ml", "weight_g", "paper_gsm"]:
        report.append(f"  {c:12s} {df[c].notna().sum()}")
    report.append(f"  paper_size   {(df['paper_size'] != 'Not stated').sum()}")
    report.append(f"  pack_count>1 {(df['pack_count'] > 1).sum()}")
    report += ["", "Grade:", df["grade"].value_counts().to_string(), "",
               "Price (Rs.) by site:", df.groupby("site")["price_lkr"].describe().round(0).to_string(), "",
               f"\nSaved: {out_path.relative_to(ROOT)}"]
    report_text = "\n".join(report)
    (OUT_DIR / "cleaning_report.txt").write_text(report_text, encoding="utf-8")
    print(report_text)


if __name__ == "__main__":
    main()