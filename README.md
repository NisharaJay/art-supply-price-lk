# art-supply-price-lk: Ensemble Learning Dashboard for Sri Lankan Art Supply Prices

CM4371 Machine Learning & Pattern Recognition: Ensemble Learning Dashboard assignment
Due: **Monday 23 Nov 2026, midnight**

## Idea
Predict and explain the prices of art supplies sold by Sri Lankan online shops. The project compares
**bagging, boosting and stacking** ensembles plus a **Random Forest + ANN hybrid**, and presents
the results in an interactive dashboard.

## Project structure
```
art-supply-price-lk/
├── README.md                  ← this file
├── AI_USAGE.md                ← declaration of AI-tool use (required by lecturer)
├── requirements.txt           ← Python libraries
├── data_collection/
│   ├── scraper.py             ← collects product data from the shops
│   └── logs/                  ← log of every scraper run (proof of collection)
├── data/
│   ├── raw/                   ← CSV exactly as collected (never edited by hand)
│   └── processed/             ← cleaned data + engineered features
├── notebooks/
│   ├── 01_cleaning_eda.ipynb          ← cleaning, feature extraction, EDA
│   └── 02_ensemble_models.ipynb       ← bagging / boosting / stacking / hybrid + comparison
├── models/                    ← trained models saved as .pkl
├── dashboard/
│   └── app.py                 ← Streamlit dashboard (loads models/*.pkl)
├── docs/
│   ├── verification.md        ← manual spot-check of scraped rows
│   └── verification_screenshots/
└── video/                     ← ≤3-minute presentation video
```

## Data sources
Collected on: **17 Nov 2026, 22:03–22:05 (Sri Lanka time)**. Raw file: `data/raw/art_supplies_raw_2026-11-17_2203.csv`
(2,181 products). Run log: `data_collection/logs/scrape_log_2026-11-17_2203.txt`

| Shop | Platform | Products | Collected? | Notes |
|---|---|---|---|---|
| haanz.lk | WooCommerce | 1,003 | ✅ | robots.txt allows product pages |
| shanscraftsstore.com | WooCommerce | 1,023 | ✅ | |
| jencilstore.com | WooCommerce | 155 | ✅ | robots.txt allows product pages |
| premiumstationers.lk | Shopify | ~737 | ❌ | Returns HTTP 429 to automated requests, so we respected that |
| artcorner.lk | WooCommerce | ~127 | ❌ | robots.txt content signal `ai-train=no` |
| nehara.lk | JavaScript app | ? | ❌ | robots.txt content signal `ai-train=no` |

Collection method: the shops' public WooCommerce Store API (`/wp-json/wc/store/v1/products`),
the same data their product pages display. There's a 2-second delay between requests, only public
product info is collected, and every row keeps its product `url` for verification.

> **Data disclaimer:** Data was collected from publicly available product listings for educational,
> non-commercial purposes. All product information belongs to the respective shops.

## How to reproduce
```bash
py -m pip install -r requirements.txt
cd data_collection
py scraper.py --max-pages 1     # quick test
py scraper.py                   # full collection → data/raw/art_supplies_raw_<date>.csv
```
