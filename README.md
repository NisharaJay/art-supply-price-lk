# ArtPrice LK: Ensemble Learning for Sri Lankan Art Supply Prices

What drives the prices of art supplies in Sri Lanka? This project predicts and explains them using
**bagging, boosting, stacking** and a **Random Forest + ANN hybrid**, with an interactive dashboard.

**Live dashboard:** https://art-supply-price-lk.streamlit.app

## Data
- **Source:** public product listings from 3 Sri Lankan online art shops: haanz.lk, shanscraftsstore.com, jencilstore.com (3 other shops were excluded. Two do not allow their data to be used for AI training and one blocked automated access)  
- **Collected:** 17 Nov 2026 with [`scraper.py`](data_collection/scraper.py) 
- **Size:** 2,181 raw products → **1,352 art supplies** after cleaning (duplicates, craft and stationery removed)
- **Verified:** 20 products checked by hand against the websites ([`docs/verification.md`](docs/verification.md))

## Results
| Model | Test R² | CV R² | Avg. error |
|---|---|---|---|
| **XGBoost (best)** | 0.570 | **0.584** | Rs. 597 |
| Stacking (RF + XGBoost + CatBoost) | 0.572 | 0.578 | Rs. 593 |
| CatBoost | 0.558 | 0.575 | Rs. 618 |
| Random Forest | 0.529 | 0.556 | Rs. 598 |

Main price drivers: pack size, sets, volume, brand (e.g. Mont Marte) and category.
Full comparison of all 12 models in [`notebooks/02_ensemble_models.ipynb`](notebooks/02_ensemble_models.ipynb).

## Project files
| Folder | Contents |
|---|---|
| `data_collection/` | Scraper and run log |
| `data_processing/` | Cleaning and feature extraction |
| `data/` | Raw and processed data |
| `notebooks/` | 01: data exploration · 02: models and comparison |
| `models/` | Saved models used by the dashboard |
| `dashboard/` | Streamlit dashboard |

## Run locally
```bash
py -m pip install -r requirements.txt
py -m streamlit run dashboard/app.py
```

## AI usage
AI tools were used and are declared in [`AI_USAGE.md`](AI_USAGE.md).

> Data was collected from publicly available listings for educational, non-commercial purposes.
> All product information belongs to the respective shops.
