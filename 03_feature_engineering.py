"""
Feature Engineering & Feature Selection (Simple)
- Create new features
- Correlation-based feature selection
- Variance check
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8-whitegrid')

# Load cleaned data
print("Loading cleaned dataset...")
df = pd.read_csv('phones_cleaned_imputed.csv')
print(f"Shape: {df.shape}\n")

# ============================================================
# FEATURE ENGINEERING
# ============================================================
print("=" * 60)
print("FEATURE ENGINEERING")
print("=" * 60)

# 1. Price_per_GB_RAM
df['Price_per_GB_RAM'] = df['Price_INR'] / df['RAM_GB']
print("1. Price_per_GB_RAM created")

# 2. Camera_Score
df['Camera_Score'] = df['Rear_Camera_MP'] + df['Front_Camera_MP']
print("2. Camera_Score created")

# 3. Value_Score
df['Value_Score'] = (df['RAM_GB'] * df['Battery_mAh']) / df['Price_INR']
print("3. Value_Score created")

# 4. Price_Tier
df['Price_Tier'] = pd.cut(df['Price_INR'],
                           bins=[0, 4000, 7000, 10000, float('inf')],
                           labels=['Budget', 'Mid', 'Premium', 'Ultra'])
print("4. Price_Tier created")

# 5. Correlation of new features with Rating
new_features = ['Price_per_GB_RAM', 'Camera_Score', 'Value_Score']
print("\nCorrelation of new features with Rating:")
for feat in new_features:
    corr = df[feat].corr(df['Rating'])
    print(f"  {feat}: {corr:.4f}")

print(f"\nDataset shape: {df.shape}")

# ============================================================
# FEATURE SELECTION (Simple: Correlation-based)
# ============================================================
print("\n" + "=" * 60)
print("FEATURE SELECTION (Correlation-based)")
print("=" * 60)

# Select numeric columns only
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
exclude = ['Product Name', 'Description', 'Price', 'Rating', 'RAM_Bucket',
           'Cluster', 'Cluster_Label']
feature_cols = [c for c in numeric_cols if c not in exclude]

X = df[feature_cols]
y = df['Rating']

# Correlation with target
print("\nFeature correlation with Rating:")
corr_with_target = X.corrwith(y).abs().sort_values(ascending=False)
print(corr_with_target.round(4))

# Keep features with correlation > 0.05
selected_features = corr_with_target[corr_with_target > 0.05].index.tolist()
print(f"\nFeatures with |corr| > 0.05: {len(selected_features)}")
print(selected_features)

# Check for high inter-feature correlation (drop one of pair if > 0.9)
corr_matrix = X[selected_features].corr().abs()
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
to_drop = [col for col in upper.columns if any(upper[col] > 0.9)]
selected_features = [f for f in selected_features if f not in to_drop]
print(f"\nAfter removing highly correlated features: {len(selected_features)}")
print(selected_features)

# Save
df.to_csv('phones_features_engineered.csv', index=False)
pd.DataFrame({'feature': selected_features}).to_csv('final_features.csv', index=False)
print("\nSaved: phones_features_engineered.csv, final_features.csv")
