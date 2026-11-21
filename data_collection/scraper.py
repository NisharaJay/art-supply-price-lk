"""
Sri Lankan Art Supply Price Scraper

AI-usage declaration:
    This script was written with the help of Claude (Anthropic, AI assistant).
    It was reviewed, understood and RUN by me. The output CSV and log were
    produced by my own run (data_collection/logs/).

Collects public product listings (name, price, category, stock) from
Sri Lankan art-supply shops via their public WooCommerce Store API,
and saves them to one CSV.

Ethics: robots.txt checked, 2 s delay between requests, public data only,
product URL kept in every row for verification.

Run:  py scraper.py              (full run)
      py scraper.py --max-pages 1  (quick test)
"""

import argparse
import csv
import html
import re
import time
from datetime import datetime
from pathlib import Path

import requests

DELAY_SECONDS = 2.0          # pause between requests (politeness)
TIMEOUT = 30                 # seconds before a request gives up
HEADERS = {"User-Agent": "Mozilla/5.0 (university coursework - price research)"}

# Shops to collect from. "platform" decides which collector function is used.
SITES = [
    {"site": "haanz.lk",            "platform": "woocommerce", "base": "https://www.haanz.lk"},
    {"site": "shanscraftsstore.com", "platform": "woocommerce", "base": "https://shanscraftsstore.com"},
    {"site": "jencilstore.com",      "platform": "woocommerce", "base": "https://jencilstore.com"},
]

# Output locations
HERE = Path(__file__).resolve().parent
RAW_DIR = HERE.parent / "data" / "raw"
LOG_DIR = HERE / "logs"

COLUMNS = [
    "site", "product_id", "name", "brand", "categories", "tags", "product_type",
    "price_lkr", "regular_price_lkr", "sale_price_lkr", "min_price_lkr", "max_price_lkr",
    "on_sale", "in_stock", "n_variants", "average_rating", "review_count",
    "sku", "short_description", "url", "scraped_at",
]

def clean_text(text, max_len=None):
    """Remove HTML tags/entities and extra spaces."""
    if not text:
        return ""
    text = re.sub(r"<[^>]+>", " ", str(text))
    text = html.unescape(text)
    text = re.sub(r"\s+", " ", text).strip()
    return text[:max_len] if max_len else text


def get_json(url, log, retries=4):
    """GET a URL and return (json, response). Waits DELAY_SECONDS afterwards.
    If the shop says "429 Too Many Requests", wait (as long as it asks) and try again."""
    for attempt in range(1, retries + 1):
        resp = requests.get(url, headers=HEADERS, timeout=TIMEOUT)
        log(f"GET {url} -> {resp.status_code}")
        if resp.status_code == 429 and attempt < retries:
            wait = int(resp.headers.get("Retry-After", 0) or 0) or 30 * attempt
            log(f"  shop asked us to slow down - waiting {wait}s (attempt {attempt}/{retries})")
            time.sleep(wait)
            continue
        time.sleep(DELAY_SECONDS)
        resp.raise_for_status()
        return resp.json(), resp


def woo_money(value, minor_unit):
    """WooCommerce gives prices in cents as strings, e.g. '625000' with minor_unit 2 -> 6250.00"""
    if value in (None, ""):
        return None
    return int(value) / (10 ** minor_unit)


# Collector 1: WooCommerce wp-json/wc/store/v1/products
def scrape_woocommerce(cfg, max_pages, log, now):
    rows, page = [], 1
    while True:
        url = f"{cfg['base']}/wp-json/wc/store/v1/products?per_page=100&page={page}"
        data, resp = get_json(url, log)
        if page == 1:
            log(f"  {cfg['site']}: shop reports {resp.headers.get('X-WP-Total', '?')} products")
        if not data:
            break
        for p in data:
            pr = p.get("prices", {})
            mu = pr.get("currency_minor_unit", 2)
            rng = pr.get("price_range") or {}
            price = woo_money(pr.get("price"), mu)
            rows.append({
                "site": cfg["site"],
                "product_id": p.get("id"),
                "name": clean_text(p.get("name")),
                "brand": "; ".join(clean_text(b.get("name")) for b in p.get("brands", []) or []),
                "categories": "; ".join(clean_text(c.get("name")) for c in p.get("categories", [])),
                "tags": "; ".join(clean_text(t.get("name")) for t in p.get("tags", [])),
                "product_type": p.get("type"),
                "price_lkr": price,
                "regular_price_lkr": woo_money(pr.get("regular_price"), mu),
                "sale_price_lkr": woo_money(pr.get("sale_price"), mu),
                "min_price_lkr": woo_money(rng.get("min_amount"), mu) if rng else price,
                "max_price_lkr": woo_money(rng.get("max_amount"), mu) if rng else price,
                "on_sale": p.get("on_sale"),
                "in_stock": p.get("is_in_stock"),
                "n_variants": len(p.get("variations", []) or []) or 1,
                "average_rating": p.get("average_rating"),
                "review_count": p.get("review_count"),
                "sku": p.get("sku"),
                "short_description": clean_text(p.get("short_description"), 300),
                "url": p.get("permalink"),
                "scraped_at": now,
            })
        log(f"  page {page}: +{len(data)} products (total {len(rows)})")
        if len(data) < 100 or (max_pages and page >= max_pages):
            break
        page += 1
    return rows


# Collector 2: Shopify products.json
def scrape_shopify(cfg, max_pages, log, now):
    rows, page = [], 1
    while True:
        url = f"{cfg['base']}/collections/{cfg['collection']}/products.json?limit=250&page={page}"
        data, _ = get_json(url, log)
        products = data.get("products", [])
        if not products:
            break
        for p in products:
            variants = p.get("variants", [])
            prices = [float(v["price"]) for v in variants if v.get("price")]
            compare = [float(v["compare_at_price"]) for v in variants if v.get("compare_at_price")]
            min_price = min(prices) if prices else None
            regular = min(compare) if compare else min_price
            rows.append({
                "site": cfg["site"],
                "product_id": p.get("id"),
                "name": clean_text(p.get("title")),
                "brand": clean_text(p.get("vendor")),
                "categories": clean_text(p.get("product_type")),
                "tags": "; ".join(p.get("tags", [])),
                "product_type": "variable" if len(variants) > 1 else "simple",
                "price_lkr": min_price,
                "regular_price_lkr": regular,
                "sale_price_lkr": min_price,
                "min_price_lkr": min_price,
                "max_price_lkr": max(prices) if prices else None,
                "on_sale": bool(compare and min_price is not None and min(compare) > min_price),
                "in_stock": any(v.get("available") for v in variants),
                "n_variants": len(variants),
                "average_rating": None,      # not provided by Shopify feed
                "review_count": None,
                "sku": variants[0].get("sku") if variants else "",
                "short_description": clean_text(p.get("body_html"), 300),
                "url": f"{cfg['base']}/products/{p.get('handle')}",
                "scraped_at": now,
            })
        log(f"  page {page}: +{len(products)} products (total {len(rows)})")
        if max_pages and page >= max_pages:
            break
        page += 1
    return rows


COLLECTORS = {"woocommerce": scrape_woocommerce, "shopify": scrape_shopify}


def main():
    parser = argparse.ArgumentParser(description="Scrape Sri Lankan art-supply prices")
    parser.add_argument("--max-pages", type=int, default=0,
                        help="limit pages per shop (0 = all). Use 1 for a quick test.")
    parser.add_argument("--out-dir", default=str(RAW_DIR), help="where to save the CSV")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    stamp = datetime.now().strftime("%Y-%m-%d_%H%M")
    now = datetime.now().isoformat(timespec="seconds")
    suffix = "_TEST" if args.max_pages else ""
    csv_path = out_dir / f"art_supplies_raw_{stamp}{suffix}.csv"
    log_path = LOG_DIR / f"scrape_log_{stamp}{suffix}.txt"
    log_file = open(log_path, "w", encoding="utf-8")

    def log(msg):
        line = f"[{datetime.now().strftime('%H:%M:%S')}] {msg}"
        print(line)
        log_file.write(line + "\n")
        log_file.flush()

    log(f"Run started. max_pages={args.max_pages or 'all'}, delay={DELAY_SECONDS}s")
    all_rows, summary = [], {}
    for cfg in SITES:
        log(f"=== {cfg['site']} ({cfg['platform']}) ===")
        try:
            rows = COLLECTORS[cfg["platform"]](cfg, args.max_pages, log, now)
        except Exception as e:  # one shop failing shouldn't stop the rest
            log(f"  ERROR on {cfg['site']}: {e}")
            rows = []
        summary[cfg["site"]] = len(rows)
        all_rows.extend(rows)

    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:  
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(all_rows)

    log("=== Summary ===")
    for site, n in summary.items():
        log(f"  {site:25s} {n:5d} products")
    log(f"  {'TOTAL':25s} {len(all_rows):5d} products")
    def short(path):  # path relative to the project, so the log does not show personal folders
        try:
            return path.resolve().relative_to(HERE.parent)
        except ValueError:
            return path.name

    log(f"CSV saved to: {short(csv_path)}")
    log(f"Log saved to: {short(log_path)}")
    log_file.close()


if __name__ == "__main__":
    main()
