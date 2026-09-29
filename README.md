# Customer Segmentation & Churn Pattern Analytics in European Banking

## Contents
| File | Purpose |
|---|---|
| `Research_Paper_Churn_Analytics.docx` | Research paper: EDA, insights, recommendations, limitations |
| `Executive_Summary.docx` | 2-page summary for government / non-technical stakeholders |
| `app.py` | Streamlit dashboard (live analytics) |
| `European_Bank.csv` | Source dataset (10,000 customers) |
| `requirements.txt` | Python dependencies |
| `charts/` | Static figures used in the documents |

## Run the dashboard
```bash
pip install -r requirements.txt
streamlit run app.py
```
Keep `European_Bank.csv` in the same folder as `app.py`, or upload the CSV from the sidebar.

## Dashboard modules
Overview | Geography | Age & Tenure | High-value explorer | Drill-down.
Sidebar filters (country, gender, age, credit band, tenure, balance segment, activity, products) update all KPIs dynamically.

## Definitions
- Age: <30, 30-45, 46-60, 60+ | Credit: <600, 600-699, 700+ | Tenure: 0-2, 3-6, 7-10 yrs
- Balance: zero, low/high split at the median non-zero balance
- High-value: balance in the top quartile (>= 127,644)
- Geographic Risk Index: country churn / overall churn x 100
- Engagement Drop Indicator: inactive churn rate minus active churn rate (percentage points)
