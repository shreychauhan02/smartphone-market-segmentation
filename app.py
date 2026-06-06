

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="Budget Phone Intelligence", layout="wide", page_icon="📱")

st.markdown("""
<style>
    .stApp { background-color: #1a1a2e; }
    .main-header { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 20px; border-radius: 10px; margin-bottom: 20px; }
    .main-header h1 { color: white; font-size: 2.5em; margin: 0; }
    .main-header p { color: #e0e0e0; font-size: 1.2em; margin: 5px 0 0 0; }
    .metric-card { background: #16213e; padding: 15px; border-radius: 8px; border-left: 4px solid #667eea; margin: 5px 0; }
    .phone-card { background: #16213e; padding: 15px; border-radius: 10px; margin: 10px 0; border: 1px solid #333; }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("""<div class="main-header">
<h1>Budget Phone Intelligence</h1>
<p>Budget Phone Intelligence — Under ₹10,000</p>
</div>""", unsafe_allow_html=True)

# Sidebar navigation
page = st.sidebar.selectbox("Navigation", [
    "Dataset Overview",
    "EDA Dashboard",
    "Phone Recommender",
    "Cluster Explorer"
])


@st.cache_data
def load_data():
    return pd.read_csv("phones_clustered.csv")


df = load_data()

# ============================================================
# PAGE 1: Dataset Overview
# ============================================================
if page == "Dataset Overview":
    st.header("Dataset Overview")
    st.write(f"Shape: {df.shape[0]} rows x {df.shape[1]} columns")

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Phones", df.shape[0])
    col2.metric("Brands", df["Brand"].nunique())
    col3.metric("Avg Rating", f"{df['Rating'].mean():.2f}")

    st.subheader("Filters")
    c1, c2, c3 = st.columns(3)
    with c1:
        brands = st.multiselect("Brand", df["Brand"].unique(), default=df["Brand"].unique()[:5])
    with c2:
        price_range = st.slider("Price Range", int(df["Price_INR"].min()), int(df["Price_INR"].max()),
                                (int(df["Price_INR"].min()), int(df["Price_INR"].max())))
    with c3:
        fiveg = st.toggle("5G Only", False)

    filtered = df[df["Brand"].isin(brands) & df["Price_INR"].between(price_range[0], price_range[1])]
    if fiveg:
        filtered = filtered[filtered["Is_5G"] == 1]

    st.dataframe(filtered, height=400)
    st.download_button("Download Filtered Data", filtered.to_csv(index=False), "filtered_phones.csv", "text/csv")

# ============================================================
# PAGE 2: EDA Dashboard
# ============================================================
elif page == "EDA Dashboard":
    st.header("EDA Dashboard")

    c1, c2 = st.columns(2)
    with c1:
        fig = px.histogram(df, x="Rating", nbins=20, title="Rating Distribution",
                           color_discrete_sequence=["#667eea"])
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        fig = px.scatter(df, x="Price_INR", y="Rating", color="Brand", title="Price vs Rating", opacity=0.6)
        st.plotly_chart(fig, use_container_width=True)

    c3, c4 = st.columns(2)
    with c3:
        avg_brand = df.groupby("Brand")["Rating"].mean().sort_values(ascending=False).head(10).reset_index()
        fig = px.bar(avg_brand, x="Rating", y="Brand", orientation="h", title="Avg Rating per Brand",
                     color="Rating", color_continuous_scale="viridis")
        st.plotly_chart(fig, use_container_width=True)
    with c4:
        fig = px.box(df, x="Is_5G", y="Rating", title="5G vs Non-5G Rating")
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Correlation Heatmap")
    numeric_cols = df.select_dtypes(include=[np.number]).columns[:8]
    corr = df[numeric_cols].corr()
    fig = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r", title="Feature Correlation")
    st.plotly_chart(fig, use_container_width=True)

# ============================================================
# PAGE 3: Phone Recommender
# ============================================================
elif page == "Phone Recommender":
    st.header("Phone Recommender")

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        budget = st.slider("Budget (₹)", 1000, 10000, 10000)
    with c2:
        min_ram = st.slider("Min RAM (GB)", 2, 8, 2)
    with c3:
        min_battery = st.slider("Min Battery (mAh)", 3000, 7000, 3000)
    with c4:
        fiveg_pref = st.checkbox("5G Required", False)

    filtered = df[(df["Price_INR"] <= budget) & (df["RAM_GB"] >= min_ram) & (df["Battery_mAh"] >= min_battery)]
    if fiveg_pref:
        filtered = filtered[filtered["Is_5G"] == 1]

    if "Value_Score" in filtered.columns:
        filtered = filtered.sort_values("Value_Score", ascending=False)

    st.subheader(f"Top {min(5, len(filtered))} Recommended Phones")
    for _, row in filtered.head(5).iterrows():
        with st.container():
            st.markdown(f"""<div class="phone-card">
            <h3>{row["Product Name"]}</h3>
            <p>₹{row["Price_INR"]:.0f} | RAM: {row["RAM_GB"]:.0f}GB | Battery: {row["Battery_mAh"]:.0f}mAh | Camera: {row["Rear_Camera_MP"]:.0f}MP | Rating: {row["Rating"]:.1f} | 5G: {"Yes" if row.get("Is_5G", 0) == 1 else "No"}</p>
            </div>""", unsafe_allow_html=True)

# ============================================================
# PAGE 4: Cluster Explorer
# ============================================================
elif page == "Cluster Explorer":
    st.header("Cluster Explorer")

    if "Cluster_Label" in df.columns:
        fig = px.scatter(df, x="Price_INR", y="Rating", color="Cluster_Label",
                         hover_data=["Product Name", "Brand"], title="Phone Clusters", opacity=0.6)
        st.plotly_chart(fig, use_container_width=True)

        selected_cluster = st.selectbox("Select Cluster", df["Cluster_Label"].unique())
        cluster_phones = df[df["Cluster_Label"] == selected_cluster]

        st.subheader(f"Cluster: {selected_cluster}")
        col1, col2, col3 = st.columns(3)
        col1.metric("Phones", len(cluster_phones))
        col2.metric("Avg Price", f"₹{cluster_phones['Price_INR'].mean():.0f}")
        col3.metric("Avg Rating", f"{cluster_phones['Rating'].mean():.2f}")

        st.dataframe(cluster_phones[["Product Name", "Brand", "Price_INR", "RAM_GB",
                                      "Battery_mAh", "Rating"]].head(10))
    else:
        st.warning("Run 05_clustering.py first to generate cluster labels.")
