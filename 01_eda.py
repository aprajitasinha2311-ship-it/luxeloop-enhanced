# =============================================================================
# LuxeLoop India – Exploratory Data Analysis (EDA)
# =============================================================================
# Beginner-friendly EDA script covering:
#   - Dataset overview
#   - Univariate analysis (one variable at a time)
#   - Bivariate analysis (two variables)
#   - Correlation heatmap
#   - Distribution of target variable
# =============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ----- 0. Setup ---------------------------------------------------------------
os.makedirs("outputs/eda", exist_ok=True)          # folder to save plots
df = pd.read_csv("LuxeLoop_India_Cleaned.csv")     # load cleaned data
print("Dataset shape:", df.shape)
print("\nFirst 3 rows:\n", df.head(3))
print("\nData types:\n", df.dtypes)
print("\nBasic statistics:\n", df.describe())

# ----- 1. Missing Values Check ------------------------------------------------
print("\n--- Missing Values ---")
missing = df.isnull().sum()
print(missing[missing > 0] if missing.sum() > 0 else "No missing values found ✓")

# ----- 2. Target Variable Distribution ----------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle("Target Variable: LuxeLoop Adoption", fontsize=14, fontweight='bold')

# Pie chart — Adoption Binary
adoption_counts = df['adoption_binary'].value_counts()
axes[0].pie(adoption_counts, labels=['Will Adopt (1)', 'Will Not Adopt (0)'],
            autopct='%1.1f%%', colors=['#2ecc71', '#e74c3c'], startangle=90)
axes[0].set_title("Adoption Binary (0/1)")

# Bar chart — Adoption Likelihood (5 levels)
likelihood_order = ['Very Likely', 'Likely', 'Neutral', 'Unlikely', 'Very Unlikely']
likelihood_counts = df['adoption_likelihood'].value_counts().reindex(likelihood_order)
axes[1].bar(likelihood_counts.index, likelihood_counts.values,
            color=['#1abc9c', '#2ecc71', '#f39c12', '#e67e22', '#e74c3c'])
axes[1].set_title("Adoption Likelihood (5-level)")
axes[1].set_xlabel("Likelihood Category")
axes[1].set_ylabel("Count")
axes[1].tick_params(axis='x', rotation=20)

plt.tight_layout()
plt.savefig("outputs/eda/01_target_distribution.png", dpi=150)
plt.show()

# ----- 3. Demographic Profiles -----------------------------------------------
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle("Demographic Distributions", fontsize=14, fontweight='bold')

demo_cols = {
    'age_group': 'Age Group',
    'gender': 'Gender',
    'city_tier': 'City Tier',
    'occupation': 'Occupation',
    'income_band': 'Monthly Income Band',
    'persona': 'Customer Persona'
}

for ax, (col, title) in zip(axes.flat, demo_cols.items()):
    counts = df[col].value_counts()
    ax.barh(counts.index, counts.values, color=sns.color_palette("Set2", len(counts)))
    ax.set_title(title)
    ax.set_xlabel("Count")
    for bar, val in zip(ax.patches, counts.values):
        ax.text(bar.get_width() + 5, bar.get_y() + bar.get_height() / 2,
                str(val), va='center', fontsize=8)

plt.tight_layout()
plt.savefig("outputs/eda/02_demographics.png", dpi=150)
plt.show()

# ----- 4. Likert Scale Distributions (Rating Variables) ----------------------
rating_cols = {
    'counterfeit_concern': 'Counterfeit Concern (1-5)',
    'auth_premium_willingness': 'Auth Premium Willingness (1-5)',
    'escrow_preference': 'Escrow Preference (1-5)',
    'adoption_score': 'Adoption Score (1-5)',
    'incumbent_platform_trust': 'Incumbent Platform Trust (1-5)',
    'brand_tier_score': 'Brand Tier Score (1-4)'
}

fig, axes = plt.subplots(2, 3, figsize=(16, 8))
fig.suptitle("Likert Scale / Rating Variables", fontsize=14, fontweight='bold')

for ax, (col, title) in zip(axes.flat, rating_cols.items()):
    sns.histplot(df[col], bins=len(df[col].unique()), ax=ax, color='steelblue', kde=False)
    ax.set_title(title)
    ax.set_xlabel("Score")
    ax.set_ylabel("Count")
    ax.axvline(df[col].mean(), color='red', linestyle='--', label=f"Mean={df[col].mean():.2f}")
    ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig("outputs/eda/03_likert_distributions.png", dpi=150)
plt.show()

# ----- 5. Adoption Rate by Demographics (Bivariate) --------------------------
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("Adoption Rate by Key Demographics", fontsize=14, fontweight='bold')

bivar_cols = ['persona', 'age_group', 'city_tier', 'income_band']

for ax, col in zip(axes.flat, bivar_cols):
    # Calculate adoption rate per category
    rates = df.groupby(col)['adoption_binary'].mean().sort_values(ascending=False)
    colors = ['#2ecc71' if r >= 0.5 else '#e74c3c' for r in rates.values]
    ax.barh(rates.index, rates.values * 100, color=colors)
    ax.set_title(f"Adoption Rate by {col.replace('_', ' ').title()}")
    ax.set_xlabel("Adoption Rate (%)")
    ax.axvline(50, color='black', linestyle='--', linewidth=0.8)
    for bar, val in zip(ax.patches, rates.values):
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                f"{val*100:.1f}%", va='center', fontsize=8)

plt.tight_layout()
plt.savefig("outputs/eda/04_adoption_by_demographics.png", dpi=150)
plt.show()

# ----- 6. Spend & Budget Distributions ----------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Financial Behavior", fontsize=14, fontweight='bold')

# Log-scale distribution of Average Spend INR
axes[0].hist(df['avg_spend_inr'], bins=30, color='#3498db', edgecolor='white')
axes[0].set_title("Avg Spend Per Purchase (INR)")
axes[0].set_xlabel("Amount (INR)")
axes[0].set_ylabel("Count")

# Annual Budget INR
axes[1].hist(df['annual_budget_inr'], bins=30, color='#9b59b6', edgecolor='white')
axes[1].set_title("Annual Luxury Budget (INR)")
axes[1].set_xlabel("Amount (INR)")
axes[1].set_ylabel("Count")

plt.tight_layout()
plt.savefig("outputs/eda/05_financial_distributions.png", dpi=150)
plt.show()

# ----- 7. Correlation Heatmap -------------------------------------------------
numeric_cols = [
    'monthly_income_inr', 'brand_tier_score', 'counterfeit_concern',
    'auth_premium_willingness', 'escrow_preference', 'avg_spend_inr',
    'annual_budget_inr', 'resale_new_pref_score', 'adoption_score',
    'research_days', 'price_switch_threshold_pct', 'incumbent_platform_trust',
    'trust_composite_score', 'spend_to_income_ratio',
    'category_diversity_score', 'feature_demand_score', 'adoption_binary'
]

corr_matrix = df[numeric_cols].corr()

plt.figure(figsize=(14, 10))
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))  # show lower triangle only
sns.heatmap(corr_matrix, mask=mask, annot=True, fmt=".2f",
            cmap='coolwarm', center=0, linewidths=0.5,
            annot_kws={'size': 7})
plt.title("Correlation Heatmap – Numeric Variables", fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig("outputs/eda/06_correlation_heatmap.png", dpi=150)
plt.show()

# ----- 8. Trust & Authentication Analysis ------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Trust & Authentication Behavior", fontsize=14, fontweight='bold')

# Trust Composite Score by Persona
trust_by_persona = df.groupby('persona')['trust_composite_score'].mean().sort_values()
axes[0].barh(trust_by_persona.index, trust_by_persona.values, color='teal')
axes[0].set_title("Avg Trust Score by Persona")
axes[0].set_xlabel("Trust Composite Score (avg)")

# Auth Premium Willingness vs Adoption Binary
willingness_adoption = df.groupby('auth_premium_willingness')['adoption_binary'].mean()
axes[1].bar(willingness_adoption.index, willingness_adoption.values * 100, color='coral')
axes[1].set_title("Auth Premium Willingness vs Adoption Rate")
axes[1].set_xlabel("Willingness Score (1-5)")
axes[1].set_ylabel("Adoption Rate (%)")

plt.tight_layout()
plt.savefig("outputs/eda/07_trust_analysis.png", dpi=150)
plt.show()

print("\n✅ EDA complete! All plots saved to outputs/eda/")
