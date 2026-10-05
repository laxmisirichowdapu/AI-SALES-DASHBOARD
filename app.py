import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LinearRegression

st.set_page_config(page_title="AI Sales Dashboard", layout="wide")


# ---------- 1. DATA ----------
@st.cache_data
def load_data():
    """Make fake sales data. Later, replace this with: pd.read_csv("data/sales.csv")"""
    rng = np.random.default_rng(42)
    dates = pd.date_range("2025-01-01", "2026-09-30", freq="D")
    prices = {"Laptop": 900, "Phone": 600, "Monitor": 250, "Headphones": 120, "Keyboard": 60}
    regions = ["North", "South", "East", "West"]
    spikes = {"2025-11-28": 3.0, "2026-03-15": 0.2, "2026-07-04": 2.5}  # unusual days

    rows = []
    for i, day in enumerate(dates):
        growth = 1 + i / 700
        weekend = 1.3 if day.dayofweek >= 5 else 1.0
        n_orders = int(rng.poisson(12 * growth * weekend * spikes.get(str(day.date()), 1)))
        for _ in range(n_orders):
            product = rng.choice(list(prices), p=[0.1, 0.2, 0.2, 0.25, 0.25])
            qty = int(rng.integers(1, 4))
            rows.append((day, product, rng.choice(regions), qty, prices[product] * qty))
    return pd.DataFrame(rows, columns=["date", "product", "region", "quantity", "revenue"])


df = load_data()

# ---------- 2. SIDEBAR FILTERS ----------
st.sidebar.header("Filters")
date_range = st.sidebar.date_input(
    "Date range", (df["date"].min().date(), df["date"].max().date())
)
regions = st.sidebar.multiselect("Region", sorted(df["region"].unique()), default=sorted(df["region"].unique()))
products = st.sidebar.multiselect("Product", sorted(df["product"].unique()), default=sorted(df["product"].unique()))

if len(date_range) != 2:
    st.info("Pick both a start date and an end date.")
    st.stop()

mask = (
    (df["date"].dt.date >= date_range[0])
    & (df["date"].dt.date <= date_range[1])
    & df["region"].isin(regions)
    & df["product"].isin(products)
)
data = df[mask]
if data.empty:
    st.warning("No data for these filters.")
    st.stop()

# ---------- 3. SUMMARY NUMBERS ----------
st.title("AI Sales Dashboard")
st.caption("Sales data, charts, forecast and AI insights in one place.")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Total revenue", f"${data['revenue'].sum():,.0f}")
c2.metric("Orders", f"{len(data):,}")
c3.metric("Average order", f"${data['revenue'].mean():,.0f}")
c4.metric("Top product", data.groupby("product")["revenue"].sum().idxmax())

# Daily revenue (used by charts, forecast and anomaly detection)
daily = data.groupby("date", as_index=False)["revenue"].sum()

# ---------- 4. CHARTS ----------
st.subheader("Sales over time")
st.plotly_chart(px.line(daily, x="date", y="revenue", labels={"revenue": "Revenue ($)", "date": ""}),
                use_container_width=True)

left, right = st.columns(2)
by_product = data.groupby("product", as_index=False)["revenue"].sum().sort_values("revenue", ascending=False)
by_region = data.groupby("region", as_index=False)["revenue"].sum()
left.subheader("Revenue by product")
left.plotly_chart(px.bar(by_product, x="product", y="revenue"), use_container_width=True)
right.subheader("Revenue by region")
right.plotly_chart(px.pie(by_region, names="region", values="revenue"), use_container_width=True)

# ---------- 5. AI: FORECAST ----------
st.subheader("AI sales forecast (next 30 days)")


def make_features(dates, start):
    """Day number (trend) + day of week (weekly pattern)."""
    X = pd.DataFrame({"t": (dates - start).days})
    for d in range(7):
        X[f"dow{d}"] = (dates.dayofweek == d).astype(int)
    return X


if len(daily) >= 30:
    start = daily["date"].min()
    model = LinearRegression().fit(make_features(pd.DatetimeIndex(daily["date"]), start), daily["revenue"])
    future_dates = pd.date_range(daily["date"].max() + pd.Timedelta(days=1), periods=30)
    forecast = pd.DataFrame({"date": future_dates, "revenue": model.predict(make_features(future_dates, start))})

    hist = daily.tail(90).assign(type="Actual")
    fut = forecast.assign(type="Forecast")
    st.plotly_chart(px.line(pd.concat([hist, fut]), x="date", y="revenue", color="type"),
                    use_container_width=True)
    st.write(f"Expected revenue for the next 30 days: **${forecast['revenue'].sum():,.0f}**")
else:
    st.info("Need at least 30 days of data to forecast.")

# ---------- 6. AI: UNUSUAL SALES ----------
st.subheader("Unusual sales days (anomaly detection)")
if len(daily) >= 30:
    iso = IsolationForest(contamination=0.01, random_state=42)
    daily["unusual"] = iso.fit_predict(daily[["revenue"]]) == -1
    fig = px.scatter(daily, x="date", y="revenue", color="unusual",
                     color_discrete_map={True: "red", False: "lightgrey"})
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(daily[daily["unusual"]][["date", "revenue"]].reset_index(drop=True))

# ---------- 7. PLAIN-ENGLISH INSIGHTS ----------
st.subheader("Insights")
last7 = daily.tail(7)["revenue"].sum()
prev7 = daily.iloc[-14:-7]["revenue"].sum() if len(daily) >= 14 else 0
if prev7 > 0:
    change = (last7 - prev7) / prev7 * 100
    word = "up" if change >= 0 else "down"
    st.write(f"- Sales are **{word} {abs(change):.1f}%** compared with the week before.")
st.write(f"- **{by_product.iloc[0]['product']}** is the best-selling product.")
st.write(f"- **{by_region.sort_values('revenue').iloc[-1]['region']}** is the strongest region, "
         f"and **{by_region.sort_values('revenue').iloc[0]['region']}** is the weakest.")
if "unusual" in daily:
    st.write(f"- **{int(daily['unusual'].sum())}** unusual sales day(s) were found. Check them for problems or special events.")