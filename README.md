# Budget Phone Intelligence - [https://smartphone-market-segmentation-hyocppkahemfuzec4faun7.streamlit.app/]

A clustering-based phone segmentation project that analyzes budget phones (under ₹10,000) scraped from Flipkart. The goal is to segment phones into meaningful groups using **KMeans clustering** so buyers can quickly find the best phone for their needs.

## Why This Project?

The Indian budget phone market is overwhelming — hundreds of models, varying specs, and no clear way to compare them. This project:

- Scrapes real phone data from Flipkart
- Cleans and engineers useful features
- Segments phones into clusters like "Value King", "Budget Workhorse", and "Camera Champion"
- Provides an interactive Streamlit app to explore and filter phones

## Pipeline

| Step | Script | What It Does |
|------|--------|-------------|
| 1 | `01_null_analysis.py` | Missing value audit, mechanism detection (MCAR/MAR/MNAR), KNN imputation |
| 2 | `02_eda.py` | Univariate, bivariate, multivariate analysis with plots |
| 3 | `03_feature_engineering.py` | Creates features (Value_Score, Camera_Score, Price_per_GB_RAM), correlation-based feature selection |
| 4 | `05_clustering.py` | KMeans clustering with elbow/silhouette analysis, PCA visualization, cluster labeling |

> Step 4 (`04_regression_models.py`) was removed — this project focuses purely on clustering, not rating prediction.

## How to Run

```bash
# Install dependencies
pip install -r requirements.txt

# Run the pipeline in order
python 01_null_analysis.py
python 02_eda.py
python 03_feature_engineering.py
python 05_clustering.py

# Launch the Streamlit app
streamlit run app.py
```

## Streamlit App Pages

1. **Dataset Overview** — Browse and filter the full phone dataset
2. **EDA Dashboard** — Interactive charts (rating distribution, price vs rating, brand analysis, correlation heatmap)
3. **Phone Recommender** — Set budget, RAM, battery preferences and get top phone recommendations
4. **Cluster Explorer** — View phone clusters, pick a cluster and see its phones

## Dataset

- **Source**: Flipkart (scraped)
- **Size**: 1,532 phones, 26 brands
- **Price Range**: ₹5,699 — ₹69,999
- **Features**: Product Name, Brand, Price, Rating, RAM, ROM, Screen Size, Rear Camera, Front Camera, Battery, 5G support

## Clusters

| Cluster | Description |
|---------|-------------|
| Value King | High RAM + battery at low price — best bang for buck |
| Budget Workhorse | Affordable phones with decent specs |
| Camera Champion | Premium phones focused on camera quality |

## Project Structure

```
project2/
├── 01_null_analysis.py        # Missing value handling
├── 02_eda.py                  # Exploratory data analysis
├── 03_feature_engineering.py  # Feature creation & selection
├── 05_clustering.py           # KMeans clustering & PCA
├── app.py                     # Streamlit web app
├── requirements.txt           # Dependencies
├── flipkart_phones_enriched.csv   # Raw scraped data
├── phones_cleaned_imputed.csv     # After null handling
├── phones_features_engineered.csv # After feature engineering
├── phones_clustered.csv           # Final clustered data
├── final_features.csv             # Selected features
├── plots/                         # Generated plots
│   ├── missing_values_visualization.png
│   ├── eda_univariate.png
│   ├── eda_bivariate.png
│   ├── eda_pairplot.png
│   ├── clustering_elbow.png
│   ├── clusters_pca.png
│   └── brand_cluster.png
└── scarping.ipynb                 # Scraping notebook
```

## Tech Stack

- **Python** — pandas, numpy, scikit-learn, matplotlib, seaborn, plotly
- **Clustering** — KMeans with silhouette-based optimal k selection
- **Dimensionality Reduction** — PCA for visualization
- **App** — Streamlit
