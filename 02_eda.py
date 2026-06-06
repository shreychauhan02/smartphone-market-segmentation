"""
Full EDA on the cleaned phones dataset.
- Univariate Analysis
- Bivariate Analysis
- Multivariate Analysis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

# Load cleaned data
print("Loading cleaned dataset...")
df = pd.read_csv('phones_cleaned_imputed.csv')
print(f"Shape: {df.shape}\n")

# ============================================================
# UNIVARIATE ANALYSIS
# ============================================================
print("=" * 60)
print("UNIVARIATE ANALYSIS")
print("=" * 60)

fig, axes = plt.subplots(2, 3, figsize=(20, 12))

axes[0, 0].hist(df['Price_INR'].dropna(), bins=30, color='steelblue', edgecolor='black')
axes[0, 0].set_title('Price_INR Distribution', fontweight='bold')
axes[0, 0].set_xlabel('Price (INR)')
axes[0, 0].axvline(df['Price_INR'].mean(), color='red', linestyle='--',
                     label=f"Mean: {df['Price_INR'].mean():.0f}")
axes[0, 0].legend()

axes[0, 1].hist(df['RAM_GB'].dropna(), bins=20, color='coral', edgecolor='black')
axes[0, 1].set_title('RAM_GB Distribution', fontweight='bold')
axes[0, 1].set_xlabel('RAM (GB)')

axes[0, 2].hist(df['Battery_mAh'].dropna(), bins=30, color='green', edgecolor='black')
axes[0, 2].set_title('Battery_mAh Distribution', fontweight='bold')
axes[0, 2].set_xlabel('Battery (mAh)')

axes[1, 0].hist(df['Rating'].dropna(), bins=20, color='purple', edgecolor='black')
axes[1, 0].set_title('Rating Distribution', fontweight='bold')
axes[1, 0].set_xlabel('Rating')

brand_counts = df['Brand'].value_counts().head(15)
axes[1, 1].barh(brand_counts.index, brand_counts.values, color='teal')
axes[1, 1].set_title('Top 15 Brands', fontweight='bold')
axes[1, 1].set_xlabel('Count')

fiveg_counts = df['Is_5G'].value_counts()
axes[1, 2].pie(fiveg_counts.values,
               labels=['Non-5G', '5G'] if 0 in fiveg_counts.index else ['5G', 'Non-5G'],
               autopct='%1.1f%%', colors=['lightcoral', 'lightskyblue'], startangle=90)
axes[1, 2].set_title('5G vs Non-5G', fontweight='bold')

plt.tight_layout()
plt.savefig('plots/eda_univariate.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved: plots/eda_univariate.png\n")

print("Observations:")
print(f"  - Price range: Rs.{df['Price_INR'].min():.0f} to Rs.{df['Price_INR'].max():.0f}, mean Rs.{df['Price_INR'].mean():.0f}")
print(f"  - Most common brand: {brand_counts.index[0]} ({brand_counts.values[0]} phones)")
print(f"  - 5G penetration: {fiveg_counts.get(1, 0) / (fiveg_counts.sum()) * 100:.1f}%\n")

# ============================================================
# BIVARIATE ANALYSIS
# ============================================================
print("=" * 60)
print("BIVARIATE ANALYSIS")
print("=" * 60)

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

top_brands = df['Brand'].value_counts().head(6).index
for brand in top_brands:
    subset = df[df['Brand'] == brand]
    axes[0, 0].scatter(subset['Price_INR'], subset['Rating'], label=brand, alpha=0.6, s=30)
axes[0, 0].set_title('Price vs Rating (colored by Brand)', fontweight='bold')
axes[0, 0].set_xlabel('Price (INR)')
axes[0, 0].set_ylabel('Rating')
axes[0, 0].legend(fontsize=8, loc='upper right')

df['RAM_Bucket'] = pd.cut(df['RAM_GB'], bins=[0, 2, 3, 4, 7], labels=['2GB', '3GB', '4GB', '6GB+'])
sns.boxplot(data=df.dropna(subset=['RAM_Bucket']), x='RAM_Bucket', y='Rating', ax=axes[0, 1], palette='Set2')
axes[0, 1].set_title('Rating by RAM Bucket', fontweight='bold')

sns.boxplot(data=df, x='Is_5G', y='Rating', ax=axes[1, 0], palette='Set3')
axes[1, 0].set_xticklabels(['Non-5G', '5G'])
axes[1, 0].set_title('5G vs Non-5G Rating', fontweight='bold')

avg_rating = df.groupby('Brand')['Rating'].mean().sort_values(ascending=False).head(15)
axes[1, 1].barh(avg_rating.index, avg_rating.values, color='steelblue')
axes[1, 1].set_title('Average Rating per Brand (Top 15)', fontweight='bold')
axes[1, 1].set_xlabel('Average Rating')
axes[1, 1].invert_yaxis()

plt.tight_layout()
plt.savefig('plots/eda_bivariate.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved: plots/eda_bivariate.png\n")

print("Observations:")
print(f"  - Price-Rating relationship is weak; budget phones can have high ratings")
print(f"  - RAM bucket {df.groupby('RAM_Bucket')['Rating'].mean().idxmax()} has the highest average rating")
print(f"  - Top rated brand: {avg_rating.index[0]} ({avg_rating.values[0]:.2f})\n")

# ============================================================
# MULTIVARIATE ANALYSIS
# ============================================================
print("=" * 60)
print("MULTIVARIATE ANALYSIS")
print("=" * 60)

numeric_data = df.select_dtypes(include=[np.number])
fig, axes = plt.subplots(1, 2, figsize=(20, 8))

corr_matrix = numeric_data.corr()
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='coolwarm', center=0, fmt='.2f',
            ax=axes[0], square=True, linewidths=0.5)
axes[0].set_title('Correlation Heatmap (Pearson)', fontweight='bold')

# Pairplot
pairplot_features = ['Price_INR', 'RAM_GB', 'Battery_mAh', 'Rating']
pairplot_data = df[pairplot_features + ['Is_5G']].dropna()
pairplot_data['Is_5G_Label'] = pairplot_data['Is_5G'].map({0: 'Non-5G', 1: '5G'})

plt.figure(figsize=(10, 8))
for i, f1 in enumerate(pairplot_features):
    for j, f2 in enumerate(pairplot_features):
        ax = plt.subplot(len(pairplot_features), len(pairplot_features), i * len(pairplot_features) + j + 1)
        if i != j:
            for label in ['Non-5G', '5G']:
                subset = pairplot_data[pairplot_data['Is_5G_Label'] == label]
                ax.scatter(subset[f2], subset[f1], alpha=0.3, s=10, label=label)
            if i == 0 and j == 1:
                ax.legend(fontsize=6)
        else:
            ax.hist(pairplot_data[f1], bins=20, alpha=0.6)
        if j == 0:
            ax.set_ylabel(f1, fontsize=8)
        if i == len(pairplot_features) - 1:
            ax.set_xlabel(f2, fontsize=8)
        ax.tick_params(labelsize=6)

plt.suptitle('Pairplot: Price, RAM, Battery, Rating by 5G', fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('plots/eda_pairplot.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved: plots/eda_pairplot.png\n")

corr_unstacked = corr_matrix.abs().unstack().sort_values(ascending=False)
corr_unstacked = corr_unstacked[corr_unstacked < 1.0]
top_pairs = corr_unstacked.head(5)
print("Top 5 Correlations:")
for idx, val in top_pairs.items():
    print(f"  {idx[0]} <-> {idx[1]}: {val:.3f}")

print("\nEDA Complete!")
