"""
STEP 1-6: Missing Value Analysis
- Null Audit
- Visualize Missing Values
- Missing Mechanism Detection (MCAR/MAR/MNAR)
- Imputation
- Confirm Zero Nulls
- Post-Imputation Distribution Check
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno
from scipy.stats import pointbiserialr, ks_2samp
from sklearn.impute import KNNImputer
import warnings
warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

# Load data
print("Loading dataset...")
df = pd.read_csv('flipkart_phones_enriched.csv')
print(f"Shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}\n")

# ============================================================
# STEP 1: NULL AUDIT
# ============================================================
print("=" * 60)
print("STEP 1: NULL AUDIT")
print("=" * 60)

null_counts = df.isnull().sum()
null_pct = (df.isnull().sum() / len(df) * 100).round(2)
null_df = pd.DataFrame({'Null Count': null_counts, 'Null %': null_pct})
null_df = null_df[null_df['Null Count'] > 0].sort_values('Null Count', ascending=False)

if len(null_df) > 0:
    print(f"\nColumns with null values ({len(null_df)} of {len(df.columns)}):")
    print(null_df.to_string())
else:
    print("\nNo null values found in any column!")

print(f"\nTotal rows: {len(df)}")
print(f"Total null cells: {df.isnull().sum().sum()}\n")

# ============================================================
# STEP 2: VISUALIZE MISSING VALUES
# ============================================================
print("=" * 60)
print("STEP 2: VISUALIZE MISSING VALUES")
print("=" * 60)

if null_df.shape[0] > 0:
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    msno.matrix(df, ax=axes[0], sparkline=False, fontsize=10)
    axes[0].set_title('Missing Value Matrix', fontsize=14, fontweight='bold')
    msno.heatmap(df, ax=axes[1], fontsize=10)
    axes[1].set_title('Missing Value Heatmap (Correlation)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('plots/missing_values_visualization.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("Saved: plots/missing_values_visualization.png")
else:
    print("No missing values to visualize!")

# ============================================================
# STEP 3: MISSING MECHANISM DETECTION
# ============================================================
print("\n" + "=" * 60)
print("STEP 3: MISSING MECHANISM DETECTION")
print("=" * 60)

numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
null_columns = null_df.index.tolist()

mechanism_results = {}

# Create null indicator columns
for col in null_columns:
    df[f'{col}_is_null'] = df[col].isnull().astype(int)

null_indicator_cols = [f'{col}_is_null' for col in null_columns]

# --- MCAR Detection ---
print("\n--- MCAR Detection (Point-Biserial Correlation) ---")
mcar_candidates = []
for null_col in null_columns:
    null_ind = df[f'{null_col}_is_null']
    other_numeric = [c for c in numeric_cols if c != null_col]

    correlations = []
    for num_col in other_numeric:
        valid = df[[num_col]].join(null_ind).dropna()
        if valid[num_col].nunique() > 1 and null_ind.sum() > 0:
            corr, pval = pointbiserialr(valid[f'{null_col}_is_null'], valid[num_col])
            correlations.append({'Feature': num_col, 'Correlation': corr, 'P-Value': pval})

    corr_df = pd.DataFrame(correlations)
    if len(corr_df) > 0:
        max_corr = corr_df['Correlation'].abs().max()
        print(f"\n{null_col}:")
        print(f"  Max |correlation| with other features: {max_corr:.4f}")
        if max_corr < 0.1:
            print(f"  -> Classified as MCAR (no significant correlation)")
            mcar_candidates.append(null_col)
        else:
            sig_corr = corr_df[corr_df['P-Value'] < 0.05]
            if len(sig_corr) > 0:
                print(f"  Significant correlations:")
                for _, row in sig_corr.iterrows():
                    print(f"    {row['Feature']}: corr={row['Correlation']:.4f}, p={row['P-Value']:.4f}")

# --- MAR Detection ---
print("\n\n--- MAR Detection (Null Correlation Heatmap) ---")
mar_candidates = []
if len(null_indicator_cols) > 1:
    null_corr = df[null_indicator_cols].corr()
    print("Null-indicator correlation matrix:")
    print(null_corr.round(3))

    plt.figure(figsize=(10, 8))
    sns.heatmap(null_corr, annot=True, cmap='RdYlBu_r', center=0, fmt='.3f')
    plt.title('Null-Indicator Correlation Heatmap (MAR Detection)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('plots/mar_heatmap.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("Saved: plots/mar_heatmap.png")

for null_col in null_columns:
    if null_col not in mcar_candidates:
        null_ind = df[f'{null_col}_is_null']
        for num_col in numeric_cols:
            if num_col != null_col:
                valid = df[[num_col]].join(null_ind).dropna()
                if len(valid) > 0 and valid[num_col].nunique() > 1:
                    corr, pval = pointbiserialr(valid[f'{null_col}_is_null'], valid[num_col])
                    if abs(corr) >= 0.1 and pval < 0.05:
                        print(f"\nMAR evidence: {null_col} nulls correlate with {num_col} (corr={corr:.4f}, p={pval:.4f})")
                        if null_col not in mar_candidates:
                            mar_candidates.append(null_col)

# --- MNAR Detection ---
print("\n\n--- MNAR Detection (KS Test on Grouped Distributions) ---")
mnar_candidates = []
for null_col in null_columns:
    if null_col not in mcar_candidates and null_col not in mar_candidates:
        null_ind = df[f'{null_col}_is_null']
        other_numeric = [c for c in numeric_cols if c != null_col]

        print(f"\n{null_col}:")
        for num_col in other_numeric[:3]:
            null_group = df.loc[null_ind == 1, num_col].dropna()
            notnull_group = df.loc[null_ind == 0, num_col].dropna()

            if len(null_group) > 0 and len(notnull_group) > 0:
                ks_stat, ks_p = ks_2samp(null_group, notnull_group)
                print(f"  {num_col}: KS stat={ks_stat:.4f}, p={ks_p:.4f}", end="")
                if ks_p < 0.05:
                    print(f" -> SIGNIFICANT (potential MNAR)")
                else:
                    print()

        if null_col not in mar_candidates:
            mnar_candidates.append(null_col)

# --- Summary Table ---
print("\n\n" + "=" * 60)
print("MISSING MECHANISM SUMMARY")
print("=" * 60)
summary = []
for col in null_columns:
    if col in mcar_candidates:
        mech = 'MCAR'
    elif col in mar_candidates:
        mech = 'MAR'
    elif col in mnar_candidates:
        mech = 'MNAR'
    else:
        mech = 'Unclassified'
    summary.append({'Column': col, 'Mechanism': mech, 'Null Count': df[col].isnull().sum()})

summary_df = pd.DataFrame(summary)
print(summary_df.to_string(index=False))

null_mechanisms = {row['Column']: row['Mechanism'] for _, row in summary_df.iterrows()}

# ============================================================
# STEP 4: IMPUTATION
# ============================================================
print("\n" + "=" * 60)
print("STEP 4: IMPUTATION")
print("=" * 60)

df_imputed = df.copy()
imputed_columns = []

for col, mech in null_mechanisms.items():
    print(f"\nImputing {col} ({mech}):")

    if mech == 'MCAR':
        median_val = df_imputed[col].median()
        df_imputed[col].fillna(median_val, inplace=True)
        print(f"  Median imputation with value: {median_val}")
        imputed_columns.append(col)

    elif mech == 'MAR':
        knn_features = [c for c in numeric_cols if c != col]
        knn_data = df_imputed[knn_features + [col]].copy()
        imputer = KNNImputer(n_neighbors=5)
        imputed_vals = imputer.fit_transform(knn_data)
        df_imputed[col] = imputed_vals[:, -1]
        print(f"  KNN Imputation (k=5) using features: {knn_features[:5]}...")
        imputed_columns.append(col)

    elif mech == 'MNAR':
        flag_name = f'{col}_was_missing'
        df_imputed[flag_name] = df_imputed[col].isnull().astype(int)
        median_val = df_imputed[col].median()
        df_imputed[col].fillna(median_val, inplace=True)
        print(f"  Added flag column: {flag_name}")
        print(f"  Then median imputation with value: {median_val}")
        imputed_columns.append(col)
        imputed_columns.append(flag_name)

# Drop null indicator columns
df_imputed.drop(columns=[f'{c}_is_null' for c in null_columns], inplace=True, errors='ignore')
print(f"\nTotal columns imputed: {len(imputed_columns)}")

# ============================================================
# STEP 5: CONFIRM ZERO NULLS
# ============================================================
print("\n" + "=" * 60)
print("STEP 5: CONFIRM ZERO NULLS")
print("=" * 60)

remaining_nulls = df_imputed.isnull().sum()
cols_with_nulls = remaining_nulls[remaining_nulls > 0]

if len(cols_with_nulls) == 0:
    print("\nSUCCESS: All columns show 0 nulls!")
else:
    print("\nWARNING: Some columns still have nulls:")
    print(cols_with_nulls)

print(f"\nTotal null cells remaining: {df_imputed.isnull().sum().sum()}")
print(f"Dataset shape: {df_imputed.shape}")

# ============================================================
# STEP 6: POST-IMPUTATION DISTRIBUTION CHECK
# ============================================================
print("\n" + "=" * 60)
print("STEP 6: POST-IMPUTATION DISTRIBUTION CHECK")
print("=" * 60)

original_imputed = [c for c in imputed_columns if '_was_missing' not in c]

for col in original_imputed:
    print(f"\n--- {col} ---")

    orig_vals = df[col].dropna()
    post_vals = df_imputed[col].dropna()

    stats_data = {
        'Metric': ['Mean', 'Std', 'Skew'],
        'Before': [orig_vals.mean(), orig_vals.std(), orig_vals.skew()],
        'After': [post_vals.mean(), post_vals.std(), post_vals.skew()]
    }
    stats_table = pd.DataFrame(stats_data).round(4)
    print(stats_table.to_string(index=False))

    ks_stat, ks_p = ks_2samp(orig_vals, post_vals)
    if ks_p < 0.05:
        print(f"WARNING: Distribution shift detected (KS p={ks_p:.4f} < 0.05)")
    else:
        print(f"OK: No significant distribution shift (KS p={ks_p:.4f})")

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    axes[0].hist(orig_vals, bins=30, alpha=0.6, label='Before Imputation', density=True, color='blue')
    axes[0].hist(post_vals, bins=30, alpha=0.6, label='After Imputation', density=True, color='red')
    axes[0].set_title(f'{col}: Histogram Overlay', fontweight='bold')
    axes[0].legend()
    axes[0].set_xlabel(col)

    axes[1].boxplot([orig_vals.values, post_vals.values], labels=['Before', 'After'])
    axes[1].set_title(f'{col}: Boxplot Comparison', fontweight='bold')
    axes[1].set_ylabel(col)

    orig_vals.plot.kde(ax=axes[2], label='Before', color='blue')
    post_vals.plot.kde(ax=axes[2], label='After', color='red')
    axes[2].set_title(f'{col}: KDE Comparison', fontweight='bold')
    axes[2].legend()
    axes[2].set_xlabel(col)

    plt.tight_layout()
    plt.savefig(f'plots/imputation_check_{col}.png', dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Saved: plots/imputation_check_{col}.png")

# Save cleaned dataset
df_imputed.to_csv('phones_cleaned_imputed.csv', index=False)
print(f"\nSaved: phones_cleaned_imputed.csv")
print(f"Final shape: {df_imputed.shape}")
