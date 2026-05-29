# =============================================================================
# LuxeLoop India – Machine Learning Models
# =============================================================================
# This script covers 4 types of analysis:
#   1. Classification  – Predict if a user will adopt LuxeLoop (0/1)
#   2. Regression      – Predict Adoption Score (1-5) or Annual Budget
#   3. Clustering      – Segment users into behaviour groups
#   4. Association Rules – Find patterns in features/categories users prefer
# =============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
warnings.filterwarnings('ignore')

os.makedirs("outputs/models", exist_ok=True)

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (classification_report, confusion_matrix,
                             mean_absolute_error, mean_squared_error, r2_score)
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.cluster import KMeans

# ─────────────────────────────────────────────────────────────────────────────
# STEP 0 — Load Data
# ─────────────────────────────────────────────────────────────────────────────
df = pd.read_csv("LuxeLoop_India_Cleaned.csv")
print("Loaded:", df.shape)

# ─────────────────────────────────────────────────────────────────────────────
# HELPER — Encode categorical columns (label encoding for tree models)
# ─────────────────────────────────────────────────────────────────────────────
def encode_df(dataframe, cat_cols):
    """Label-encode a list of categorical columns in a copy of the dataframe."""
    df_enc = dataframe.copy()
    le = LabelEncoder()
    for col in cat_cols:
        df_enc[col] = le.fit_transform(df_enc[col].astype(str))
    return df_enc


# ─────────────────────────────────────────────────────────────────────────────
# 1. CLASSIFICATION — Predict adoption_binary (0 = No, 1 = Yes)
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("1. CLASSIFICATION — Predict Adoption (0/1)")
print("="*60)

# Features selected for classification
clf_features = [
    'persona', 'age_group', 'gender', 'city_tier', 'occupation',
    'brand_tier_score', 'resale_experience',
    'counterfeit_concern', 'auth_premium_willingness', 'escrow_preference',
    'resale_new_pref_score', 'monthly_income_inr', 'avg_spend_inr',
    'trust_composite_score', 'feature_demand_score', 'category_diversity_score',
    'research_days', 'price_switch_threshold_pct', 'incumbent_platform_trust',
    'festive_purchase'
]
clf_target = 'adoption_binary'

# Encode categorical features
cat_cols_clf = ['persona', 'age_group', 'gender', 'city_tier', 'occupation',
                'resale_experience', 'festive_purchase']
df_clf = encode_df(df, cat_cols_clf)

X_clf = df_clf[clf_features]
y_clf = df_clf[clf_target]

# Train / test split (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(
    X_clf, y_clf, test_size=0.2, random_state=42, stratify=y_clf
)

# --- Model A: Logistic Regression (simple, interpretable baseline) ---
log_reg = LogisticRegression(max_iter=1000, random_state=42)
log_reg.fit(X_train, y_train)
y_pred_lr = log_reg.predict(X_test)

print("\n--- Logistic Regression ---")
print(classification_report(y_test, y_pred_lr))

# --- Model B: Random Forest (usually more accurate for tabular data) ---
rf_clf = RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42)
rf_clf.fit(X_train, y_train)
y_pred_rf = rf_clf.predict(X_test)

print("\n--- Random Forest Classifier ---")
print(classification_report(y_test, y_pred_rf))

# Confusion Matrix for Random Forest
cm = confusion_matrix(y_test, y_pred_rf)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['No Adopt', 'Adopt'],
            yticklabels=['No Adopt', 'Adopt'])
plt.title("Confusion Matrix – Random Forest Classifier")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig("outputs/models/clf_confusion_matrix.png", dpi=150)
plt.show()

# Feature Importance
feat_imp = pd.Series(rf_clf.feature_importances_, index=clf_features).sort_values(ascending=False)
plt.figure(figsize=(10, 6))
feat_imp.head(12).plot(kind='barh', color='steelblue')
plt.title("Top 12 Feature Importances – Classification")
plt.xlabel("Importance Score")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig("outputs/models/clf_feature_importance.png", dpi=150)
plt.show()
print("\nTop 5 features for predicting adoption:")
print(feat_imp.head(5))


# ─────────────────────────────────────────────────────────────────────────────
# 2. REGRESSION — Predict annual_budget_inr (continuous)
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("2. REGRESSION — Predict Annual Luxury Budget (INR)")
print("="*60)

# Features for regression
reg_features = [
    'monthly_income_inr', 'brand_tier_score', 'avg_spend_inr',
    'resale_new_pref_score', 'adoption_score', 'trust_composite_score',
    'category_diversity_score', 'feature_demand_score',
    'counterfeit_concern', 'auth_premium_willingness',
    'spend_to_income_ratio', 'research_days',
    'price_switch_threshold_pct', 'incumbent_platform_trust'
]
reg_target = 'annual_budget_inr'

cat_cols_reg = []   # all numeric features here — no encoding needed
X_reg = df[reg_features]
y_reg = df[reg_target]

X_tr, X_te, y_tr, y_te = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)

# --- Model A: Linear Regression ---
lin_reg = LinearRegression()
lin_reg.fit(X_tr, y_tr)
y_pred_lin = lin_reg.predict(X_te)

print("\n--- Linear Regression ---")
print(f"  MAE  : ₹{mean_absolute_error(y_te, y_pred_lin):,.0f}")
print(f"  RMSE : ₹{np.sqrt(mean_squared_error(y_te, y_pred_lin)):,.0f}")
print(f"  R²   : {r2_score(y_te, y_pred_lin):.4f}")

# --- Model B: Random Forest Regressor ---
rf_reg = RandomForestRegressor(n_estimators=100, max_depth=8, random_state=42)
rf_reg.fit(X_tr, y_tr)
y_pred_rfr = rf_reg.predict(X_te)

print("\n--- Random Forest Regressor ---")
print(f"  MAE  : ₹{mean_absolute_error(y_te, y_pred_rfr):,.0f}")
print(f"  RMSE : ₹{np.sqrt(mean_squared_error(y_te, y_pred_rfr)):,.0f}")
print(f"  R²   : {r2_score(y_te, y_pred_rfr):.4f}")

# Actual vs Predicted plot
plt.figure(figsize=(8, 6))
plt.scatter(y_te, y_pred_rfr, alpha=0.3, color='purple', s=10)
plt.plot([y_te.min(), y_te.max()], [y_te.min(), y_te.max()], 'r--', linewidth=1)
plt.title("Actual vs Predicted – Annual Budget (Random Forest)")
plt.xlabel("Actual Annual Budget (INR)")
plt.ylabel("Predicted Annual Budget (INR)")
plt.tight_layout()
plt.savefig("outputs/models/reg_actual_vs_predicted.png", dpi=150)
plt.show()

# Feature importances for regression
reg_feat_imp = pd.Series(rf_reg.feature_importances_, index=reg_features).sort_values(ascending=False)
plt.figure(figsize=(10, 5))
reg_feat_imp.plot(kind='barh', color='mediumpurple')
plt.title("Feature Importances – Regression")
plt.xlabel("Importance Score")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig("outputs/models/reg_feature_importance.png", dpi=150)
plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# 3. CLUSTERING — K-Means Customer Segmentation
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("3. CLUSTERING — K-Means Customer Segmentation")
print("="*60)

# Features for clustering (behavioural + financial signals)
cluster_features = [
    'monthly_income_inr', 'avg_spend_inr', 'brand_tier_score',
    'trust_composite_score', 'feature_demand_score',
    'category_diversity_score', 'adoption_score',
    'resale_new_pref_score', 'research_days',
    'price_switch_threshold_pct'
]

X_cluster = df[cluster_features].copy()

# Scale features (K-Means is distance-based — scaling is important!)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_cluster)

# --- Find optimal K using Elbow Method ---
inertias = []
k_range = range(2, 10)
for k in k_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(X_scaled)
    inertias.append(km.inertia_)

plt.figure(figsize=(8, 5))
plt.plot(k_range, inertias, 'bo-')
plt.title("Elbow Method – Optimal Number of Clusters")
plt.xlabel("Number of Clusters (k)")
plt.ylabel("Within-Cluster Sum of Squares (Inertia)")
plt.xticks(k_range)
plt.tight_layout()
plt.savefig("outputs/models/cluster_elbow.png", dpi=150)
plt.show()

# --- Fit K-Means with K=4 (good balance for luxury personas) ---
K_OPTIMAL = 4
kmeans = KMeans(n_clusters=K_OPTIMAL, random_state=42, n_init=10)
df['cluster'] = kmeans.fit_predict(X_scaled)

# Cluster profiles
cluster_profile = df.groupby('cluster')[cluster_features + ['adoption_binary']].mean().round(2)
print("\nCluster Profiles (averages):\n")
print(cluster_profile)

# Visualise clusters (2D using first 2 features as proxy)
plt.figure(figsize=(10, 6))
scatter = plt.scatter(df['avg_spend_inr'], df['trust_composite_score'],
                      c=df['cluster'], cmap='tab10', alpha=0.5, s=15)
plt.colorbar(scatter, label="Cluster ID")
plt.title(f"K-Means Clusters (K={K_OPTIMAL}) – Spend vs Trust")
plt.xlabel("Avg Spend Per Purchase (INR)")
plt.ylabel("Trust Composite Score")
plt.tight_layout()
plt.savefig("outputs/models/cluster_scatter.png", dpi=150)
plt.show()

# Cluster size distribution
plt.figure(figsize=(6, 4))
df['cluster'].value_counts().sort_index().plot(kind='bar', color='teal', edgecolor='white')
plt.title("Cluster Size Distribution")
plt.xlabel("Cluster")
plt.ylabel("Number of Respondents")
plt.tight_layout()
plt.savefig("outputs/models/cluster_sizes.png", dpi=150)
plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# 4. ASSOCIATION RULE MINING — Feature/Category Co-occurrence
# ─────────────────────────────────────────────────────────────────────────────
# NOTE: mlxtend library required: pip install mlxtend
# If not installed, this section will print a helpful message and skip.
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("4. ASSOCIATION RULE MINING")
print("="*60)

try:
    from mlxtend.frequent_patterns import apriori, association_rules
    from mlxtend.preprocessing import TransactionEncoder

    # --- Basket: each row = one respondent's desired features ---
    # Split multi-select "desired_features" column into a list
    transactions = df['desired_features'].fillna('').apply(
        lambda x: [item.strip() for item in x.split(',') if item.strip()]
    ).tolist()

    # Also add preferred categories as a second basket
    transactions_cat = df['preferred_categories'].fillna('').apply(
        lambda x: [item.strip() for item in x.split(',') if item.strip()]
    ).tolist()

    # Encode features into one-hot transaction matrix
    te = TransactionEncoder()
    te_array = te.fit_transform(transactions)
    df_basket = pd.DataFrame(te_array, columns=te.columns_)

    # Remove very rare items (appear in < 5% of transactions)
    df_basket = df_basket.loc[:, df_basket.mean() > 0.05]

    # Run Apriori algorithm
    frequent_itemsets = apriori(df_basket, min_support=0.1, use_colnames=True)
    rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1.2)
    rules = rules.sort_values("lift", ascending=False)

    print(f"\nFrequent Itemsets found: {len(frequent_itemsets)}")
    print(f"Association Rules found:  {len(rules)}")
    print("\nTop 10 Rules (by Lift):\n")
    print(rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']].head(10).to_string())

    # Save rules to CSV
    rules_export = rules.copy()
    rules_export['antecedents'] = rules_export['antecedents'].apply(lambda x: ', '.join(list(x)))
    rules_export['consequents'] = rules_export['consequents'].apply(lambda x: ', '.join(list(x)))
    rules_export.to_csv("outputs/models/association_rules.csv", index=False)
    print("\n✅ Association rules saved to outputs/models/association_rules.csv")

    # Plot top 10 rules by lift
    top_rules = rules.head(10).copy()
    top_rules['rule_label'] = (top_rules['antecedents'].apply(lambda x: ', '.join(list(x)))
                                + " → "
                                + top_rules['consequents'].apply(lambda x: ', '.join(list(x))))
    plt.figure(figsize=(12, 6))
    plt.barh(top_rules['rule_label'], top_rules['lift'], color='coral')
    plt.xlabel("Lift")
    plt.title("Top 10 Association Rules by Lift (Desired Features)")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig("outputs/models/association_rules_plot.png", dpi=150)
    plt.show()

except ImportError:
    print("\n⚠️  mlxtend is not installed. Run: pip install mlxtend")
    print("Then re-run this section.")
    print("\nManual Association Mining (frequency counts as fallback):")
    # Fallback: count most common desired features
    all_features = df['desired_features'].fillna('').str.split(',').explode().str.strip()
    top_features = all_features.value_counts().head(10)
    print(top_features)


print("\n✅ All models complete! Outputs saved to outputs/models/")
