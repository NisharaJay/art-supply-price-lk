"""
ArtPrice LK: Ensemble Learning Dashboard for Sri Lankan Art Supply Prices

AI-assisted: written with the help of Claude. SeeAI_USAGE.md.
Run locally:  py -m streamlit run dashboard/app.py
"""

from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap
import streamlit as st
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "processed" / "art_supplies_clean.csv"
MODELS_DIR = ROOT / "models"

CAT_COLS = ["category_group", "brand_grouped", "grade", "paper_size", "site"]
NUM_COLS = ["volume_ml", "weight_g", "paper_gsm", "pack_count", "is_set",
            "n_variants", "is_variable", "on_sale", "in_stock", "name_word_count"]

st.set_page_config(page_title="ArtPrice LK", page_icon="🎨", layout="wide")


# Load data and models
@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)


@st.cache_data
def load_results():
    return pd.read_csv(MODELS_DIR / "model_results.csv")


MODEL_NAMES = {  # class name inside the saved pipeline -> readable name
    "XGBRegressor": "XGBoost", "CatBoostRegressor": "CatBoost", "LGBMRegressor": "LightGBM",
    "RandomForestRegressor": "Random Forest", "BaggingRegressor": "Bagging (Trees)",
    "StackingRegressor": "Stacking (RF+XGB+Cat)", "GradientBoostingRegressor": "Gradient Boosting",
}


@st.cache_resource
def load_models():
    """The best model (whichever Notebook 2 saved) plus Random Forest and XGBoost for comparison."""
    best = joblib.load(MODELS_DIR / "best_model.pkl")
    best_name = MODEL_NAMES.get(type(best.named_steps["model"]).__name__, "Best model")
    models = {f"{best_name} (best)": best}
    for name, file in [("Random Forest", "random_forest.pkl"), ("XGBoost", "xgboost.pkl")]:
        if name != best_name:
            models[name] = joblib.load(MODELS_DIR / file)
    return models, best_name


def model_inputs(data):
    """Same feature preparation as Notebook 2."""
    X = data[CAT_COLS + NUM_COLS].copy()
    X[NUM_COLS] = X[NUM_COLS].fillna(0)
    return X


@st.cache_resource
def rf_shap_values(_rf, data):
    """SHAP values of the Random Forest on the same test set as Notebook 2."""
    X = model_inputs(data)
    _, X_test = train_test_split(X, test_size=0.2, random_state=42)
    X_prep = _rf.named_steps["prep"].transform(X_test)
    if hasattr(X_prep, "toarray"):
        X_prep = X_prep.toarray()
    X_prep = pd.DataFrame(X_prep, columns=_rf.named_steps["prep"].get_feature_names_out())
    explainer = shap.TreeExplainer(_rf.named_steps["model"])
    return explainer.shap_values(X_prep), X_prep


df = load_data()
results = load_results()
models, BEST_NAME = load_models()


# Header and sidebar
st.title("ArtPrice LK")
st.markdown("#### What drives the prices of art supplies in Sri Lanka?")
st.caption("Ensemble learning (bagging, boosting, stacking and a hybrid RF + ANN) on "
           f"{len(df):,} art supplies collected from {df['site'].nunique()} Sri Lankan online shops ")

with st.sidebar:
    st.header("About")
    st.markdown(
        "**Data:** haanz.lk, shanscraftsstore.com, jencilstore.com  \n"
        "**Collected:** 17 Nov 2026 (public product listings)  \n"
        "**Target:** log(1 + price), shown in Rs.  \n"
        f"**Best model:** {BEST_NAME}"
    )
    st.divider()

tab_data, tab_models, tab_explain, tab_predict = st.tabs(
    ["Data overview", "Model comparison", "What drives price?", "Price predictor"])


# Tab 1: Data overview
with tab_data:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Products", f"{len(df):,}")
    c2.metric("Shops", df["site"].nunique())
    c3.metric("Known brands", (df["brand_grouped"] != "Unknown").sum())
    c4.metric("Median price", f"Rs. {df['price_lkr'].median():,.0f}")

    left, right = st.columns(2)
    with left:
        st.subheader("Price by category")
        order = df.groupby("category_group")["price_lkr"].median().sort_values().index
        fig, ax = plt.subplots(figsize=(7, 6))
        sns.boxplot(data=df, x="price_lkr", y="category_group", order=order, ax=ax)
        ax.set_xscale("log")
        ax.set_xlabel("Price (Rs., log scale)")
        ax.set_ylabel("")
        st.pyplot(fig)

    with right:
        st.subheader("Price by shop")
        fig, ax = plt.subplots(figsize=(7, 4))
        sns.boxplot(data=df, x="site", y="price_lkr", ax=ax)
        ax.set_yscale("log")
        ax.set_ylabel("Price (Rs., log scale)")
        ax.set_xlabel("")
        st.pyplot(fig)
        st.dataframe(df.groupby("site")["price_lkr"].agg(["count", "median", "mean"])
                     .round(0).rename(columns={"count": "Products", "median": "Median (Rs.)",
                                               "mean": "Mean (Rs.)"}))

    st.subheader("Browse the data")
    pick = st.multiselect("Category", sorted(df["category_group"].unique()))
    view = df[df["category_group"].isin(pick)] if pick else df
    st.dataframe(view[["name", "site", "category_group", "brand", "price_lkr", "url"]]
                 .sort_values("price_lkr"), width="stretch", hide_index=True)


# Tab 2: Model comparison
with tab_models:
    st.subheader("All models on the same test set")
    st.dataframe(results, width="stretch", hide_index=True)

    left, right = st.columns(2)
    with left:
        fig, ax = plt.subplots(figsize=(7, 6))
        sns.barplot(data=results, x="Test R2", y="Model", hue="Type", dodge=False, ax=ax)
        ax.set_title("Test R² (higher is better)")
        ax.set_ylabel("")
        st.pyplot(fig)
    with right:
        fig, ax = plt.subplots(figsize=(7, 6))
        sns.barplot(data=results, x="MAE (Rs.)", y="Model", hue="Type", dodge=False, ax=ax)
        ax.set_title("Average error in Rs. (lower is better)")
        ax.set_ylabel("")
        st.pyplot(fig)

    st.subheader("Overfitting check: train vs test")
    gap = results.dropna(subset=["Train R2"]).melt(id_vars="Model", value_vars=["Train R2", "Test R2"],
                                                   var_name="Set", value_name="R2")
    fig, ax = plt.subplots(figsize=(12, 4))
    sns.barplot(data=gap, x="Model", y="R2", hue="Set", ax=ax)
    ax.set_xlabel("")
    plt.xticks(rotation=30, ha="right")
    st.pyplot(fig)
    st.caption("A big gap between train and test (e.g. the single Decision Tree) means overfitting. "
               "Ensembles reduce this gap.")


# Tab 3: What drives price?
with tab_explain:
    rf = models.get("Random Forest") or models["Random Forest (best)"]
    left, right = st.columns(2)

    with left:
        st.subheader("Feature importance (Random Forest)")
        names = rf.named_steps["prep"].get_feature_names_out()
        importance = (pd.Series(rf.named_steps["model"].feature_importances_, index=names)
                      .sort_values(ascending=False).head(15))
        importance.index = [n.split("__", 1)[1] for n in importance.index]
        fig, ax = plt.subplots(figsize=(7, 6))
        importance.sort_values().plot(kind="barh", ax=ax)
        ax.set_xlabel("Importance")
        st.pyplot(fig)

    with right:
        st.subheader("SHAP (Random Forest)")
        with st.spinner("Calculating SHAP values..."):
            shap_values, X_prep = rf_shap_values(rf, df)
        X_show = X_prep.copy()
        X_show.columns = [c.split("__", 1)[1] for c in X_show.columns]
        plt.figure()
        shap.summary_plot(shap_values, X_show, max_display=15, show=False)
        st.pyplot(plt.gcf())

    st.info("**How to read SHAP:** each dot is a product. Red = high feature value, blue = low. "
            "Dots on the right increase the predicted price. Sets, bigger packs and Mont Marte "
            "products increase the price; Fabric & Glass paints decrease it.")


# Tab 4: Price predictor
with tab_predict:
    PAINT_LIKE = {"Acrylic Paint", "Oil Paint", "Watercolour & Gouache", "Poster & Tempera Paint",
                  "Fabric & Glass Paint", "Mediums & Varnishes", "Pens & Markers"}
    best_mae = results.loc[results["Model"] == BEST_NAME, "MAE (Rs.)"]

    def typical(series, default=0):
        """Most typical (median) value of a size column, used as a sensible default."""
        s = series.dropna()
        return int(round(s.median())) if len(s) else default

    def show_predictions(item):
        preds = {name: float(np.expm1(m.predict(item)[0])) for name, m in models.items()}
        cols = st.columns(len(preds))
        for col, (name, price) in zip(cols, preds.items()):
            col.metric(name, f"Rs. {price:,.0f}")
        return preds

    mode = st.radio("Mode", ["Describe a product", "Test on a real product"], horizontal=True)

    # Mode A: describe a product 
    if mode == "Describe a product":
        st.subheader("Estimate the price of an art supply")
        c1, c2, c3 = st.columns(3)
        with c1:
            category = st.selectbox("Category", df["category_group"].value_counts().index)
            in_cat = df[df["category_group"] == category]
            # only brands that exist in this category, most common first
            brand = st.selectbox("Brand", in_cat["brand_grouped"].value_counts().index)
            shop = st.selectbox("Shop", sorted(df["site"].unique()))
        similar = in_cat[in_cat["brand_grouped"] == brand]
        with c2:
            grade = st.selectbox("Grade", ["Unspecified", "Professional", "Student/Kids"])
            pack_count = st.number_input("Number of items (1 = single item, e.g. 24 for a 24-colour set)",
                                         min_value=1, max_value=150,
                                         value=typical(similar["pack_count"], 1))
        with c3:
            volume_ml = weight_g = paper_gsm = 0
            paper_size = "Not stated"
            if category in PAINT_LIKE:
                volume_ml = st.number_input("Volume per tube/bottle (ml)", min_value=0, max_value=1000,
                                            value=typical(similar["volume_ml"], typical(in_cat["volume_ml"])))
            if category == "Paper & Canvas":
                sizes = ["Not stated", "A5", "A4", "A3", "A2", "A1"]
                paper_size = st.selectbox("Paper size", sizes)
                paper_gsm = st.number_input("Paper thickness (gsm), 0 if not known", min_value=0,
                                            max_value=500, value=0)
            if category == "Clay & Sculpting":
                weight_g = st.number_input("Weight (g)", min_value=0, max_value=1000,
                                           value=typical(similar["weight_g"], typical(in_cat["weight_g"])))

        item = pd.DataFrame([{
            "category_group": category, "brand_grouped": brand, "grade": grade,
            "paper_size": paper_size, "site": shop,
            "volume_ml": volume_ml, "weight_g": weight_g, "paper_gsm": paper_gsm,
            "pack_count": pack_count, "is_set": int(pack_count > 1), "n_variants": 1,
            "is_variable": 0, "on_sale": 0, "in_stock": 1,
            # not something a user can describe, so use the typical value for this brand/category
            "name_word_count": typical((similar if len(similar) else in_cat)["name_word_count"], 5),
        }])

        if st.button("Predict price", type="primary"):
            show_predictions(item)
            if len(best_mae):
                st.caption(f"Average error of the best model: about Rs. {best_mae.iloc[0]:,.0f}. "
                           "Prices depend on details a listing does not show, so treat this as a rough estimate.")
            ref = similar if len(similar) else in_cat
            label = "unknown-brand" if brand == "Unknown" else brand
            st.markdown(f"**Real prices of {label} products in {category}:** Rs. {ref['price_lkr'].min():,.0f} – "
                        f"Rs. {ref['price_lkr'].max():,.0f} (median Rs. {ref['price_lkr'].median():,.0f}, "
                        f"{len(ref)} products)")
            # most similar real products: same number of items and closest size first
            ref = ref.assign(_pack=(ref["pack_count"] - pack_count).abs(),
                             _vol=(ref["volume_ml"].fillna(0) - volume_ml).abs())
            st.dataframe(ref.sort_values(["_pack", "_vol"]).head(8)[["name", "site", "price_lkr", "url"]],
                         width="stretch", hide_index=True)

    # Mode B: test on a real product
    else:
        st.subheader("How close are the models on products they have never seen?")
        # Same 80/20 split as Notebook 2: only the 20% test products are offered here, because
        # testing on products a model was trained on would make it look better than it really is.
        _, X_test = train_test_split(model_inputs(df), test_size=0.2, random_state=42)
        test_df = df.loc[X_test.index].sort_values("name")
        st.caption(f"These {len(test_df)} products were kept aside and NOT used to train the models.")

        # overall comparison on all unseen products
        errors = {}
        for name, m in models.items():
            pred = np.expm1(m.predict(model_inputs(test_df)))
            errors[name] = np.abs(pred - test_df["price_lkr"].values) / test_df["price_lkr"].values
        cols = st.columns(len(errors))
        for col, (name, e) in zip(cols, errors.items()):
            col.metric(f"{name}: typical error", f"{np.median(e) * 100:.0f}%",
                       help="Median % difference from the real price over all unseen products")
        names = list(errors)
        if len(names) == 2:
            closer = (errors[names[1]] < errors[names[0]]).mean() * 100
            st.caption(f"{names[1]} is closer on {closer:.0f}% of these products and {names[0]} on "
                       f"{100 - closer:.0f}%. Random Forest is often slightly closer on typical products, "
                       f"while {BEST_NAME} makes fewer large errors (higher R²).")

        st.markdown("**Try one product**")
        choice = st.selectbox("Product", test_df["name"] + "  |  " + test_df["site"])
        name, site = choice.split("  |  ")
        product = test_df[(test_df["name"] == name) & (test_df["site"] == site)].iloc[[0]]
        real = product["price_lkr"].iloc[0]
        st.metric("Real price", f"Rs. {real:,.0f}")
        preds = show_predictions(model_inputs(product))
        st.caption(" | ".join(f"{n}: {abs(p - real) / real * 100:.0f}% {'above' if p > real else 'below'}"
                              for n, p in preds.items())
                   + f"  ·  [Open product page]({product['url'].iloc[0]})")