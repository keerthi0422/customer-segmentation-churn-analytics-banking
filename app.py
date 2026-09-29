"""
Customer Segmentation & Churn Pattern Analytics in European Banking
Streamlit dashboard  -  run with:  streamlit run app.py
"""
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="European Banking Churn Analytics", page_icon="🏦", layout="wide")

NAVY, RED, TEAL, GREY = "#1F3A5F", "#C0392B", "#2A9D8F", "#95A5A6"
AGE_ORDER = ["<30", "30-45", "46-60", "60+"]
CREDIT_ORDER = ["Low (<600)", "Medium (600-699)", "High (700+)"]
TENURE_ORDER = ["New (0-2 yrs)", "Mid-term (3-6 yrs)", "Long-term (7-10 yrs)"]
BAL_ORDER = ["Zero balance", "Low balance", "High balance"]


# ----------------------------------------------------------------------------
# 1. Data ingestion, validation, cleaning, segmentation
# ----------------------------------------------------------------------------
def validate(df: pd.DataFrame) -> list[str]:
    """Return a list of data-quality messages (empty list = all checks passed)."""
    issues = []
    required = ["CustomerId", "CreditScore", "Geography", "Gender", "Age", "Tenure", "Balance",
                "NumOfProducts", "HasCrCard", "IsActiveMember", "EstimatedSalary", "Exited"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        return [f"Missing required columns: {missing}"]
    if df[required].isna().any().any():
        issues.append("Null values found in required columns.")
    if df.CustomerId.duplicated().any():
        issues.append(f"{int(df.CustomerId.duplicated().sum())} duplicate CustomerId values.")
    for c in ["HasCrCard", "IsActiveMember", "Exited"]:
        if not set(df[c].unique()) <= {0, 1}:
            issues.append(f"{c} is not strictly binary (0/1).")
    if not (df.NumOfProducts >= 1).all():
        issues.append("NumOfProducts contains values below 1.")
    if not set(df.Geography.unique()) <= {"France", "Spain", "Germany"}:
        issues.append("Unexpected Geography values.")
    return issues


@st.cache_data(show_spinner=False)
def prepare(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw.drop(columns=[c for c in ["Surname", "Year"] if c in raw.columns]).copy()
    df["AgeGroup"] = pd.cut(df.Age, [0, 29, 45, 60, 200], labels=AGE_ORDER)
    df["CreditBand"] = pd.cut(df.CreditScore, [0, 599, 699, 1000], labels=CREDIT_ORDER)
    df["TenureGroup"] = pd.cut(df.Tenure, [-1, 2, 6, 10], labels=TENURE_ORDER)
    med = df.loc[df.Balance > 0, "Balance"].median()
    df["BalanceSegment"] = np.where(df.Balance == 0, "Zero balance",
                                    np.where(df.Balance <= med, "Low balance", "High balance"))
    df["BalanceSegment"] = pd.Categorical(df.BalanceSegment, BAL_ORDER)
    df["Activity"] = df.IsActiveMember.map({1: "Active", 0: "Inactive"})
    return df


def rate(s: pd.Series) -> float:
    return float(s.mean() * 100) if len(s) else float("nan")


def seg_table(d: pd.DataFrame, col: str, total_churn: int) -> pd.DataFrame:
    g = d.groupby(col, observed=True).agg(Customers=("Exited", "size"), Churned=("Exited", "sum"),
                                          ChurnRate=("Exited", lambda x: x.mean() * 100)).reset_index()
    g["ShareOfCustomers"] = g.Customers / max(len(d), 1) * 100
    g["ShareOfChurn"] = g.Churned / max(total_churn, 1) * 100
    return g


# ----------------------------------------------------------------------------
# 2. Load data
# ----------------------------------------------------------------------------
st.sidebar.title("🏦 Churn Analytics")
uploaded = st.sidebar.file_uploader("Upload dataset (optional)", type="csv")
try:
    raw = pd.read_csv(uploaded) if uploaded else pd.read_csv("European_Bank.csv")
except FileNotFoundError:
    st.error("Place European_Bank.csv next to app.py or upload it in the sidebar.")
    st.stop()

problems = validate(raw)
if problems:
    st.sidebar.warning("Data validation:\n\n- " + "\n- ".join(problems))
else:
    st.sidebar.success(f"Data validated: {len(raw):,} customers, no issues")
df_all = prepare(raw)
overall_all = rate(df_all.Exited)

# ----------------------------------------------------------------------------
# 3. Sidebar segment filters
# ----------------------------------------------------------------------------
st.sidebar.header("Segment filters")
geo = st.sidebar.multiselect("Geography", ["France", "Germany", "Spain"], default=["France", "Germany", "Spain"])
gen = st.sidebar.multiselect("Gender", ["Female", "Male"], default=["Female", "Male"])
age = st.sidebar.multiselect("Age group", AGE_ORDER, default=AGE_ORDER)
cred = st.sidebar.multiselect("Credit score band", CREDIT_ORDER, default=CREDIT_ORDER)
ten = st.sidebar.multiselect("Tenure group", TENURE_ORDER, default=TENURE_ORDER)
bal = st.sidebar.multiselect("Balance segment", BAL_ORDER, default=BAL_ORDER)
act = st.sidebar.multiselect("Activity", ["Active", "Inactive"], default=["Active", "Inactive"])
prod = st.sidebar.multiselect("Number of products", [1, 2, 3, 4], default=[1, 2, 3, 4])

df = df_all[df_all.Geography.isin(geo) & df_all.Gender.isin(gen) & df_all.AgeGroup.isin(age)
            & df_all.CreditBand.isin(cred) & df_all.TenureGroup.isin(ten)
            & df_all.BalanceSegment.isin(bal) & df_all.Activity.isin(act) & df_all.NumOfProducts.isin(prod)]

st.title("Customer Segmentation & Churn Pattern Analytics")
st.caption("European retail banking  |  France, Germany, Spain  |  10,000 customers")

if df.empty:
    st.warning("No customers match the current filters. Widen the filters in the sidebar.")
    st.stop()

total_churn = int(df.Exited.sum())
overall = rate(df.Exited)

# ----------------------------------------------------------------------------
# 4. KPIs (dynamic - respond to filters)
# ----------------------------------------------------------------------------
hv_cut = df_all.Balance.quantile(0.75)
hv = df[df.Balance >= hv_cut]
inactive, active = df[df.IsActiveMember == 0], df[df.IsActiveMember == 1]
eng_gap = rate(inactive.Exited) - rate(active.Exited) if len(inactive) and len(active) else float("nan")
bal_lost = df.loc[df.Exited == 1, "Balance"].sum()

k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Overall churn rate", f"{overall:.1f}%", f"{overall - overall_all:+.1f} pp vs. full base", delta_color="inverse")
k2.metric("Customers churned", f"{total_churn:,}", f"of {len(df):,} in view", delta_color="off")
k3.metric("High-value churn ratio", f"{rate(hv.Exited):.1f}%" if len(hv) else "n/a",
          help=f"Churn among customers with balance >= {hv_cut:,.0f} (top 25% of all customers).")
k4.metric("Engagement drop indicator", f"{eng_gap:+.1f} pp" if not np.isnan(eng_gap) else "n/a",
          help="Churn rate of inactive members minus churn rate of active members.")
k5.metric("Balance walked out", f"{bal_lost/1e6:,.1f}M",
          help="Sum of account balances held by churned customers in view (currency units as in dataset).")

tabs = st.tabs(["📊 Overview", "🌍 Geography", "👥 Age & Tenure", "💎 High-value explorer", "🔎 Drill-down"])

# ---- Overview
with tabs[0]:
    c1, c2 = st.columns(2)
    with c1:
        pie = df.Exited.map({1: "Churned", 0: "Retained"}).value_counts().reset_index()
        pie.columns = ["Status", "Customers"]
        fig = px.pie(pie, names="Status", values="Customers", hole=0.55, color="Status",
                     color_discrete_map={"Churned": RED, "Retained": NAVY}, title="Churned vs retained")
        st.plotly_chart(fig, width="stretch")
    with c2:
        dims = {"Geography": "Geography", "Gender": "Gender", "Age group": "AgeGroup", "Credit band": "CreditBand",
                "Tenure group": "TenureGroup", "Balance segment": "BalanceSegment", "Activity": "Activity",
                "Products": "NumOfProducts"}
        rows = []
        for label, col in dims.items():
            t = seg_table(df, col, total_churn)
            for _, r in t.iterrows():
                rows.append({"Dimension": label, "Segment": str(r[col]), "Churn rate (%)": r.ChurnRate, "Customers": r.Customers})
        allseg = pd.DataFrame(rows)
        allseg = allseg[allseg.Customers >= 30].sort_values("Churn rate (%)", ascending=False).head(10)
        allseg["Label"] = allseg.Dimension + ": " + allseg.Segment
        fig = px.bar(allseg.sort_values("Churn rate (%)"), x="Churn rate (%)", y="Label", orientation="h",
                     title="Ten highest-churn segments (min. 30 customers)", color_discrete_sequence=[RED],
                     hover_data=["Customers"])
        fig.add_vline(x=overall, line_dash="dash", line_color=GREY, annotation_text="overall")
        st.plotly_chart(fig, width="stretch")

    st.subheader("Churned vs retained profile")
    prof = df.groupby(df.Exited.map({1: "Churned", 0: "Retained"})).agg(
        Customers=("Exited", "size"), AvgAge=("Age", "mean"), AvgCreditScore=("CreditScore", "mean"),
        AvgTenure=("Tenure", "mean"), AvgBalance=("Balance", "mean"), AvgSalary=("EstimatedSalary", "mean"),
        AvgProducts=("NumOfProducts", "mean"), ActiveShare=("IsActiveMember", lambda x: x.mean() * 100)).round(1)
    st.dataframe(prof, width="stretch")

# ---- Geography
with tabs[1]:
    g = seg_table(df, "Geography", total_churn)
    g["Geographic risk index"] = g.ChurnRate / overall * 100
    c1, c2 = st.columns(2)
    with c1:
        fig = px.bar(g, x="Geography", y="ChurnRate", text=g.ChurnRate.round(1).astype(str) + "%",
                     title="Churn rate by country", color="ChurnRate", color_continuous_scale="Reds")
        fig.update_layout(coloraxis_showscale=False, yaxis_title="Churn rate (%)")
        st.plotly_chart(fig, width="stretch")
    with c2:
        m = g.melt(id_vars="Geography", value_vars=["ShareOfCustomers", "ShareOfChurn"], var_name="Measure", value_name="%")
        m["Measure"] = m.Measure.map({"ShareOfCustomers": "Share of customers", "ShareOfChurn": "Share of churners"})
        fig = px.bar(m, x="Geography", y="%", color="Measure", barmode="group", title="Customer base vs churn contribution",
                     color_discrete_sequence=[GREY, RED])
        st.plotly_chart(fig, width="stretch")
    st.markdown("**Geographic Risk Index** = country churn rate / overall churn rate x 100 (100 = average risk).")
    st.dataframe(g.rename(columns={"ChurnRate": "Churn rate (%)", "ShareOfCustomers": "% of customers",
                                   "ShareOfChurn": "% of churners"}).round(1), width="stretch", hide_index=True)
    st.subheader("Geography x age interaction")
    h = df.pivot_table(index="Geography", columns="AgeGroup", values="Exited", aggfunc="mean", observed=True) * 100
    st.plotly_chart(px.imshow(h.round(1), text_auto=True, color_continuous_scale="Reds", aspect="auto",
                              title="Churn % by country and age group"), width="stretch")

# ---- Age & tenure
with tabs[2]:
    c1, c2 = st.columns(2)
    for col, order, title, holder in [("AgeGroup", AGE_ORDER, "Churn rate by age group", c1),
                                      ("TenureGroup", TENURE_ORDER, "Churn rate by tenure group", c2)]:
        t = seg_table(df, col, total_churn)
        fig = px.bar(t, x=col, y="ChurnRate", text=t.ChurnRate.round(1).astype(str) + "%", title=title,
                     category_orders={col: order}, color="ChurnRate", color_continuous_scale="Reds",
                     hover_data=["Customers", "ShareOfChurn"])
        fig.add_hline(y=overall, line_dash="dash", line_color=GREY)
        fig.update_layout(coloraxis_showscale=False, yaxis_title="Churn rate (%)", xaxis_title="")
        holder.plotly_chart(fig, width="stretch")
    c3, c4 = st.columns(2)
    with c3:
        aa = df.groupby(["AgeGroup", "Activity"], observed=True).Exited.mean().mul(100).reset_index()
        fig = px.bar(aa, x="AgeGroup", y="Exited", color="Activity", barmode="group", category_orders={"AgeGroup": AGE_ORDER},
                     title="Age group x activity", color_discrete_map={"Active": TEAL, "Inactive": RED},
                     labels={"Exited": "Churn rate (%)"})
        st.plotly_chart(fig, width="stretch")
    with c4:
        ty = df.groupby("Tenure").Exited.mean().mul(100).reset_index()
        fig = px.line(ty, x="Tenure", y="Exited", markers=True, title="Churn rate by tenure year", labels={"Exited": "Churn rate (%)"})
        fig.update_traces(line_color=NAVY)
        st.plotly_chart(fig, width="stretch")
    st.plotly_chart(px.histogram(df.assign(Status=df.Exited.map({1: "Churned", 0: "Retained"})), x="Age", color="Status",
                                 barmode="overlay", nbins=40, opacity=0.65, histnorm="percent", title="Age distribution",
                                 color_discrete_map={"Churned": RED, "Retained": NAVY}), width="stretch")

# ---- High-value explorer
with tabs[3]:
    st.markdown("Define *high-value* by balance. The default is the top quartile of all customers.")
    thr = st.slider("Minimum balance for 'high-value'", 0, int(df_all.Balance.max()), int(hv_cut), step=5000)
    hvx = df[df.Balance >= thr]
    if hvx.empty:
        st.info("No customers above this balance under the current filters.")
    else:
        churned_hv = hvx[hvx.Exited == 1]
        a, b, c, d = st.columns(4)
        a.metric("High-value customers", f"{len(hvx):,}")
        b.metric("High-value churn rate", f"{rate(hvx.Exited):.1f}%", f"{rate(hvx.Exited) - overall:+.1f} pp vs. view", delta_color="inverse")
        c.metric("Balance at risk (churned)", f"{churned_hv.Balance.sum()/1e6:,.1f}M")
        d.metric("Share of all churned balance", f"{churned_hv.Balance.sum() / max(bal_lost, 1) * 100:.1f}%")
        c1, c2 = st.columns(2)
        with c1:
            sample = df.sample(min(len(df), 3000), random_state=1).assign(Status=lambda x: x.Exited.map({1: "Churned", 0: "Retained"}))
            fig = px.scatter(sample, x="EstimatedSalary", y="Balance", color="Status", opacity=0.55,
                             color_discrete_map={"Churned": RED, "Retained": NAVY}, title="Salary vs balance (sample of up to 3,000)")
            fig.add_hline(y=thr, line_dash="dash", line_color=GREY)
            st.plotly_chart(fig, width="stretch")
        with c2:
            q = df.assign(B=np.where(df.Balance >= df.Balance.median(), "High balance", "Low balance"),
                          S=np.where(df.EstimatedSalary >= df.EstimatedSalary.median(), "High salary", "Low salary"))
            qq = q.pivot_table(index="B", columns="S", values="Exited", aggfunc="mean") * 100
            st.plotly_chart(px.imshow(qq.round(1), text_auto=True, color_continuous_scale="Reds", title="Churn %: salary vs balance (median splits)"),
                            width="stretch")
        st.subheader("High-value churn by country and activity")
        hg = hvx.groupby(["Geography", "Activity"], observed=True).Exited.agg(["size", "mean"]).reset_index()
        hg["Churn rate (%)"] = hg["mean"] * 100
        st.plotly_chart(px.bar(hg, x="Geography", y="Churn rate (%)", color="Activity", barmode="group",
                               color_discrete_map={"Active": TEAL, "Inactive": RED}, hover_data=["size"]), width="stretch")
        st.subheader("Highest-balance churned customers")
        st.dataframe(churned_hv.sort_values("Balance", ascending=False).head(25)[
            ["CustomerId", "Geography", "Gender", "Age", "Tenure", "Balance", "EstimatedSalary", "NumOfProducts", "IsActiveMember"]],
            width="stretch", hide_index=True)

# ---- Drill-down
with tabs[4]:
    d1, d2 = st.columns(2)
    dim_map = {"Geography": "Geography", "Gender": "Gender", "Age group": "AgeGroup", "Credit band": "CreditBand",
               "Tenure group": "TenureGroup", "Balance segment": "BalanceSegment", "Activity": "Activity", "Products": "NumOfProducts"}
    rowd = d1.selectbox("Rows", list(dim_map), index=0)
    cold = d2.selectbox("Columns", list(dim_map), index=2)
    if rowd == cold:
        st.info("Choose two different dimensions.")
    else:
        pv = df.pivot_table(index=dim_map[rowd], columns=dim_map[cold], values="Exited", aggfunc="mean", observed=True) * 100
        cnt = df.pivot_table(index=dim_map[rowd], columns=dim_map[cold], values="Exited", aggfunc="size", observed=True)
        st.plotly_chart(px.imshow(pv.round(1), text_auto=True, color_continuous_scale="Reds", aspect="auto",
                                  title=f"Churn % - {rowd} x {cold}"), width="stretch")
        st.caption("Cell counts (small cells are noisy - treat cells with fewer than ~50 customers with caution):")
        st.dataframe(cnt, width="stretch")
    st.subheader("Segment table")
    dim = st.selectbox("Segment by", list(dim_map), index=2, key="segdim")
    st.dataframe(seg_table(df, dim_map[dim], total_churn).rename(columns={
        "ChurnRate": "Churn rate (%)", "ShareOfCustomers": "% of customers", "ShareOfChurn": "% of churners"}).round(1),
        width="stretch", hide_index=True)
    st.download_button("Download filtered data (CSV)", df.to_csv(index=False).encode(), "filtered_customers.csv", "text/csv")
