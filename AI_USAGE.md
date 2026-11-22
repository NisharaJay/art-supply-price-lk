# AI Usage Declaration

As permitted, generative AI tools were used in this project. Every use is declared below.

| Part | AI tool | How it was used | My own work |
|---|---|---|---|
| `data_collection/scraper.py` | Claude | Wrote the scraper code | Chose 6 shops I have bought art supplies from and I know, checked each site's rules and selected 3, reviewed and ran the script and verified the data by hand (`docs/verification.md`) |
| `data_processing/clean_and_features.py` | Claude | Wrote the cleaning and feature-extraction code | Decided the scope (art supplies only, not craft), reviewed the category groups and brand list, spotted mislabelled products and had them fixed, and checked the outputs (`data/processed/cleaning_report.txt`) |
| Notebooks | Claude | Provided the steps and templates for both notebooks, explained the concepts and helped fix errors | Wrote code with the help of the Lecture and Lab notebooks, ran all analyses and models, checked the results, wrote all the interpretations, comparisons and conclusions, found that Random Forest was often closer than XGBoost on individual products and investigated why |
| Dashboard (`dashboard/app.py`) | Claude | Wrote the Streamlit dashboard code (front end) | Decided the dashboard content and tested it |

The modelling approach follows the Lecture and Lab practicals: train/test split, Random Forest, Gradient Boosting, stacking, feature importance and SHAP, and regression metrics (Lab Sessions and Lecture Notebooks). The AI helped to adapt these techniques to regression and to write code for parts that were not implemented in the course notebooks (Example: Pipeline/ColumnTransformer and the ANN).

The model comparisons, results interpretation and conclusions are my own and written by myself.
