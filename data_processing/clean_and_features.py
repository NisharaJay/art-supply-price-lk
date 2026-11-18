"""
Clean the raw scraped data and build model features.

AI-usage declaration:
    Written with the help of Claude. Reviewed and run by me.

Input : data/raw/art_supplies_raw_*.csv  
Output: data/processed/art_supplies_clean.csv
        data/processed/cleaning_report.txt

Run:  py clean_and_features.py
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
    r"diamond painting|tape runner|glue applicator|silicone cup"
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
    ("Pencils, Charcoal & Pastels", r"pencil|charcoal|pastel(?!\s*paint)|crayon|graphite"),
    ("Mediums & Varnishes",       r"medium|varnish|gesso|fixative|modell?ing paste|linseed|turpentine"),
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
    (r"faber[ -]?castell", "Faber-Castell"),
    (r"fabriano", "Fabriano"),
    (r"pentel", "Pentel"),
    (r"sakura|pigma|micron", "Sakura"),
    (r"camlin|camel|fabrica", "Camlin (Camel)"),
    (r"bomeijia", "Bomeijia"),
    (r"staedtler", "Staedtler"),
    (r"tombow?\b", "Tombow"),
    (r"posca|\buni\b", "Uni"),
    (r"prismacolor", "Prismacolor"),
    (r"giorgione", "Giorgione"),
    (r"rotring", "Rotring"),
    (r"nyoni", "Nyoni"),
    (r"fujin", "Fujin"),
    (r"macron", "Macron"),
    (r"winsor|newton", "Winsor & Newton"),
    (r"\breeves\b", "Reeves"),
    (r"derwent", "Derwent"),
    (r"talens|rembrandt|van gogh", "Royal Talens"),
    (r"superior", "Superior"),
    (r"keep smiling", "Keep Smiling"),
    (r"stabilo", "Stabilo"),
    (r"nataraj", "Nataraj"),
    (r"\bapple\b", "Apple (brushes)"),
    (r"\bflash\b", "Flash"),
    (r"hobby ?craft", "Hobby Craft"),
]
MIN_BRAND_COUNT = 10   # brands with fewer products are grouped as "Other brand"


def find_category(text):
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
        r"set of (\d+)",
        r"(\d+)\s*-?\s*(?:pcs|pc|pieces|colou?rs|shades|sheets|sticks|pens|pencils|brushes|markers)\b",
        r"(\d+)\s*(?:set|pack)\b",
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

    df["category_group"] = text.apply(find_category)
    df["brand"] = [find_brand(b, t) for b, t in zip(df["brand"].fillna(""), text)]
    counts = df["brand"].value_counts()
    df["brand_grouped"] = df["brand"].where(
        df["brand"].map(counts).ge(MIN_BRAND_COUNT) | df["brand"].eq("Unknown"), "Other brand")

    df["volume_ml"] = name.apply(lambda t: first_number(t, r"(\d+(?:\.\d+)?)\s*ml\b"))
    df["weight_g"] = name.apply(extract_weight_g)
    df["paper_gsm"] = name.apply(lambda t: first_number(t, r"(\d+)\s*gsm"))
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
