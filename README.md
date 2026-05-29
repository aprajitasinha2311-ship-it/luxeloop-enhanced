# 💎 LuxeLoop India — Analytics Dashboard v2.0

> Full-stack analytics dashboard built on 2,000-respondent survey of Indian luxury pre-owned market consumers.

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-app-name.streamlit.app)

---

## 📋 Project Overview

**LuxeLoop** is a premium platform for authenticated pre-owned luxury goods in India. This v2.0 dashboard includes:

- **Exploratory Data Analysis** — Demographics, behaviour, trust, correlations
- **Classification** — Random Forest adoption predictor (ROC-AUC ~0.87)
- **Regression** — Annual budget prediction (R² ~0.82)
- **K-Means Clustering** — Behavioural segmentation with PCA, radar charts, elbow
- **Association Rule Mining** — Feature co-occurrence (Apriori)
- **Campaign Strategy** — 3-tier action plan, revenue projections, do's & don'ts
- **AI Purchase Predictor** — Real-time probability with personalised insights
- **AI Analyst Chat** — Claude-powered Q&A on all findings (with offline fallback)
- **Business Recommendations** — 7 data-driven strategic recommendations

---

## 🗂️ Repository Structure

```
luxeloop-analytics/
├── app.py                        # Streamlit dashboard (main entry point)
├── 01_eda.py                     # Standalone EDA script
├── 02_models.py                  # Standalone ML models script
├── requirements.txt              # Python dependencies
├── README.md                     # This file
└── LuxeLoop_India_Cleaned.csv    # Dataset (2,000 rows × 46 columns)
```

---

## 🚀 Quick Start (Local)

```bash
git clone https://github.com/YOUR_USERNAME/luxeloop-analytics.git
cd luxeloop-analytics
pip install -r requirements.txt
streamlit run app.py
```

App opens at **http://localhost:8501**

---

## ☁️ Deploy on Streamlit Community Cloud

1. Push repo to GitHub (public repository)
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Click **New app** → select repo → branch: `main` → main file: `app.py`
4. Click **Deploy**

> ⚠️ Make sure `LuxeLoop_India_Cleaned.csv` is committed to the repo.

### Optional: Add Anthropic API Key (for AI Chat)

In Streamlit Cloud → your app → **Settings → Secrets**, add:

```toml
# Not required — the app works without this (offline fallback mode)
# The AI Chat section calls the Anthropic API directly from the browser
```

The AI Analyst Chat works in **offline fallback mode** without any API key, providing pre-written expert answers to common questions.

---

## 📊 Dashboard Sections

| # | Section | Description |
|---|---------|-------------|
| 1 | 🏠 Project Overview | Dataset preview, variable mapping |
| 2 | 📊 Descriptive Analytics | Demographics, trust, adoption drivers, correlations |
| 3 | 🎯 Classification | RF adoption predictor + ROC curve + feature importance |
| 4 | 📈 Regression | Annual budget model + residuals |
| 5 | 🔵 Clustering | K-Means + PCA + radar chart + elbow |
| 6 | 🔗 Association Rules | Apriori mining + scatter plot |
| 7 | 📣 Campaign Strategy | 3-tier plan + revenue projection + do's & don'ts |
| 8 | 🤖 AI Purchase Predictor | Customer profile → probability + insights |
| 9 | 💬 AI Analyst Chat | Claude-powered Q&A with fallback |
| 10 | 💡 Business Recommendations | 7 strategic recommendations + adoption funnel |

---

## 📦 Dependencies

| Library | Purpose |
|---------|---------|
| `streamlit` | Interactive web dashboard |
| `pandas` / `numpy` | Data manipulation |
| `matplotlib` / `seaborn` | Visualisations |
| `scikit-learn` | ML models + clustering |
| `mlxtend` | Apriori association rules |
| `requests` | Claude API calls (AI Chat) |

---

## 💡 Key Findings

- **54.8% adoption intent** among surveyed consumers
- **Trust is the #1 conversion driver** (composite score 3.77/5)
- **Tier 2 cities** show comparable adoption at lower CAC
- **Festive season** represents a 54.7% purchase-intent spike
- **K=4 optimal clusters**: Budget Explorer, Mid-Market Buyer, Affluent Aspirant, HNW Collector
- **Escrow + AI Auth** are the most co-demanded platform features

---

*Built with ❤️ for the LuxeLoop India Analytics project · v2.0*
