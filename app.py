import io
import os

import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LinearRegression

st.set_page_config(page_title="AI Sales Dashboard", layout="wide")
DEFAULT_PATH = "data/sales.csv"


# ---------- 1. LOAD KAGGLE DATA ----------
@st.cache_data
def read_csv(source):
    """source = file path (str) or uploaded file bytes. Kaggle files are often latin-1 encoded."""
    def open_src():
        return io.BytesIO(source) if isinstance(source, bytes) else source
    try:
        return pd.read_csv(open_src())
    except UnicodeDecodeError:
        return pd.read_csv(open_src(), encoding="latin-1")


def guess(cols, keywords):
    for k in keywords:
        for c in cols:
            if k in c.lower():
                return c
    return None


st.sidebar.header("Data")
uploaded = st.sidebar.file_uploader("Upload Kaggle sales CSV", type="csv")
if uploaded is not None:
    raw = read_csv(uploaded.getvalue())
elif os.path.exists(DEFAULT_PATH):
    raw = read_csv(DEFAULT_PATH)
else:
    st.title("AI Sales Dashboard")
    st.info("Upload a sales CSV in the sidebar, or save it as data/sales.csv in your project folder.")
    st.stop()

cols = list(raw.columns)
st.sidebar.subheader("Match your columns")
date_guess, rev_guess = guess(cols, ["date"]), guess(cols, ["sales", "revenue", "amount", "total"])
prod_guess, reg_guess = guess(cols, ["category", "product", "item"]), guess(cols, ["region", "state", "city", "country"])

date_col = st.sidebar.selectbox("Date column", cols, index=cols.index(date_guess) if date_guess else 0)
rev_col = st.sidebar.selectbox("Sales amount column", cols, index=cols.index(rev_guess) if rev_guess else 0)
prod_col = st.sidebar.selectbox("Product column", cols, index=cols.index(prod_guess) if prod_guess else 0)
reg_options = ["(none)"] + cols
reg_col = st.sidebar.selectbox("Region column (optional)", reg_options,
                               index=reg_options.index(reg_guess) if reg_guess else 0)
day_first = st.sidebar.checkbox("Dates are day/month/year", value=False)

df = pd.DataFrame({
    "date": pd.to_datetime(raw[date_col], errors="coerce", dayfirst=day_first),
    "revenue": pd.to_numeric(raw[rev_col], errors="coerce"),
    "product": raw[prod_col].astype(str),
    "region": raw[reg_col].astype(str) if reg_col != "(none)" else "All",
}).dropna(subset=["date", "revenue"])

if df.empty:
    st.error("No usable rows. Check that the date and sales columns are chosen correctly.")
    st.stop()

# ---------- 2. FILTERS ----------
st.sidebar.header("Filters")
date_range = st.sidebar.date_input("Date range", (df["date"].min().date(), df["date"].max().date()))
region_list = sorted(df["region"].unique())
regions = st.sidebar.multiselect("Region", region_list, default=region_list) if reg_col != "(none)" else region_list

if len(date_range) != 2:
    st.info("Pick both a start date and an end date.")
    st.stop()

data = df[(df["date"].dt.date >= date_range[0]) & (df["date"].dt.date <= date_range[1])
          & df["region"].isin(regions)]
if data.empty:
    st.warning("No data for these filters.")
    st.stop()

# ---------- 3. SUMMARY NUMBERS ----------
st.title("AI Sales Dashboard")
st.caption("Real sales data, charts, forecast and AI insights in one place.")

by_product = data.groupby("product", as_index=False)["revenue"].sum().sort_values("revenue", ascending=False)
c1, c2, c3, c4 = st.columns(4)
c1.metric("Total sales", f"${data['revenue'].sum():,.0f}")
c2.metric("Orders (rows)", f"{len(data):,}")
c3.metric("Average order", f"${data['revenue'].mean():,.0f}")
c4.metric("Top product", by_product.iloc[0]["product"])

daily = data.groupby("date", as_index=False)["revenue"].sum()

# ---------- 4. CHARTS ----------
st.subheader("Sales over time")
st.plotly_chart(px.line(daily, x="date", y="revenue"), use_container_width=True)

left, right = st.columns(2)
left.subheader("Top 10 products")
left.plotly_chart(px.bar(by_product.head(10), x="revenue", y="product", orientation="h"),
                  use_container_width=True)
by_region = data.groupby("region", as_index=False)["revenue"].sum().sort_values("revenue", ascending=False)
if reg_col != "(none)":
    right.subheader("Sales by region (top 10)")
    right.plotly_chart(px.pie(by_region.head(10), names="region", values="revenue"), use_container_width=True)

# ---------- 5. AI: FORECAST ----------
st.subheader("AI sales forecast (next 30 days)")


def make_features(dates, start):
    X = pd.DataFrame({"t": (dates - start).days})
    for d in range(7):
        X[f"dow{d}"] = (dates.dayofweek == d).astype(int)
    return X


if len(daily) >= 30:
    start = daily["date"].min()
    model = LinearRegression().fit(make_features(pd.DatetimeIndex(daily["date"]), start), daily["revenue"])
    future = pd.date_range(daily["date"].max() + pd.Timedelta(days=1), periods=30)
    forecast = pd.DataFrame({"date": future, "revenue": model.predict(make_features(future, start))})
    plot_df = pd.concat([daily.tail(90).assign(type="Actual"), forecast.assign(type="Forecast")])
    st.plotly_chart(px.line(plot_df, x="date", y="revenue", color="type"), use_container_width=True)
    st.write(f"Expected sales for the next 30 days: **${forecast['revenue'].sum():,.0f}**")
else:
    st.info("Need at least 30 days with sales to forecast.")

# ---------- 6. AI: UNUSUAL SALES ----------
st.subheader("Unusual sales days (anomaly detection)")
if len(daily) >= 30:
    daily["unusual"] = IsolationForest(contamination=0.01, random_state=42).fit_predict(daily[["revenue"]]) == -1
    st.plotly_chart(px.scatter(daily, x="date", y="revenue", color="unusual",
                               color_discrete_map={True: "red", False: "lightgrey"}), use_container_width=True)
    st.dataframe(daily[daily["unusual"]][["date", "revenue"]].reset_index(drop=True))

# ---------- 7. INSIGHTS ----------
st.subheader("Insights")
last_day = daily["date"].max()
recent = daily[daily["date"] > last_day - pd.Timedelta(days=30)]["revenue"].sum()
before = daily[(daily["date"] <= last_day - pd.Timedelta(days=30))
               & (daily["date"] > last_day - pd.Timedelta(days=60))]["revenue"].sum()
if before > 0:
    change = (recent - before) / before * 100
    st.write(f"- Sales in the last 30 days are **{'up' if change >= 0 else 'down'} {abs(change):.1f}%** "
             f"compared with the 30 days before.")
st.write(f"- **{by_product.iloc[0]['product']}** is the best-selling product.")
if reg_col != "(none)" and len(by_region) > 1:
    st.write(f"- **{by_region.iloc[0]['region']}** is the strongest region, "
             f"and **{by_region.iloc[-1]['region']}** is the weakest.")
if "unusual" in daily:
    st.write(f"- **{int(daily['unusual'].sum())}** unusual sales day(s) found. Check them for problems or special events.")