import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# --- Page Setup ---
st.set_page_config(page_title="Predictive Delivery Optimizer", layout="wide")

st.title("🚚 Predictive Delivery Optimizer - NexGen Logistics")
st.markdown("### Analyze and predict delivery delays across orders, routes, and carriers")

# --- Load Data ---
@st.cache_data
def load_data():
    # ✅ Corrected CSV filenames
    orders = pd.read_csv("orders.csv")
    delivery = pd.read_csv("delivery_performance.csv")
    vehicles = pd.read_csv("vehicle_fleet.csv")
    costs = pd.read_csv("cost_breakdown.csv")

    # Debugging info
    print("Orders columns:", orders.columns)
    print("Delivery columns:", delivery.columns)

    # ✅ Merge using correct key (assuming both have 'Order_ID')
    df = orders.merge(delivery, on="Order_ID", how="left")

    # ✅ Compute delay metrics
    if "Actual_Delivery_Days" in df.columns and "Promised_Delivery_Days" in df.columns:
        df["Delay Days"] = df["Actual_Delivery_Days"] - df["Promised_Delivery_Days"]
        df["Delay Flag"] = np.where(df["Delay Days"] > 0, 1, 0)
    else:
        st.warning("⚠️ Delivery day columns not found, skipping delay calculation.")

    return df, vehicles, costs


# --- Load the data ---
df, vehicles, costs = load_data()

# --- Sidebar Filters ---
st.sidebar.header("Filter Data")
if "Priority" in df.columns:
    priority = st.sidebar.multiselect("Select Priority:", df["Priority"].dropna().unique(), default=df["Priority"].dropna().unique())
else:
    priority = []

if "Product Category" in df.columns:
    category = st.sidebar.multiselect("Select Category:", df["Product Category"].dropna().unique(), default=df["Product Category"].dropna().unique())
else:
    category = []

if len(priority) > 0 and len(category) > 0:
    filtered_df = df[(df["Priority"].isin(priority)) & (df["Product Category"].isin(category))]
else:
    filtered_df = df.copy()

# --- KPI Metrics ---
st.subheader("📈 Key Performance Indicators")
col1, col2, col3, col4 = st.columns(4)

total_orders = len(filtered_df)
delayed_orders = filtered_df["Delay Flag"].sum() if "Delay Flag" in filtered_df.columns else 0
avg_delay = round(filtered_df["Delay Days"].mean(), 2) if "Delay Days" in filtered_df.columns else 0
avg_rating = round(filtered_df["Customer Rating"].mean(), 2) if "Customer Rating" in filtered_df.columns else 0

col1.metric("Total Orders", total_orders)
col2.metric("Delayed Deliveries", delayed_orders)
col3.metric("Avg Delay (days)", avg_delay)
col4.metric("Avg Rating", avg_rating)

st.markdown("---")

# --- Visualization Section ---
st.subheader("📊 Delay Analysis")

# 1️⃣ Delay % by Priority
if "Priority" in filtered_df.columns and "Delay Flag" in filtered_df.columns:
    delay_by_priority = filtered_df.groupby("Priority")["Delay Flag"].mean() * 100
    fig1, ax1 = plt.subplots()
    sns.barplot(x=delay_by_priority.index, y=delay_by_priority.values, palette="coolwarm", ax=ax1)
    ax1.set_title("Delay Percentage by Priority")
    ax1.set_ylabel("Delay %")
    st.pyplot(fig1)

# 2️⃣ Delay Trend Over Time
if "Order Date" in filtered_df.columns and "Delay Flag" in filtered_df.columns:
    filtered_df["Order Date"] = pd.to_datetime(filtered_df["Order Date"], errors="coerce")
    delay_trend = filtered_df.groupby(filtered_df["Order Date"].dt.date)["Delay Flag"].mean() * 100
    fig2, ax2 = plt.subplots()
    delay_trend.plot(ax=ax2)
    ax2.set_title("Delay Trend Over Time")
    ax2.set_ylabel("Delay %")
    st.pyplot(fig2)

# 3️⃣ Scatter: Distance vs Delay
if "Distance (km)" in filtered_df.columns and "Delay Days" in filtered_df.columns:
    fig3, ax3 = plt.subplots()
    sns.scatterplot(data=filtered_df, x="Distance (km)", y="Delay Days", hue="Priority" if "Priority" in filtered_df.columns else None, ax=ax3)
    ax3.set_title("Distance vs Delay Days")
    st.pyplot(fig3)

# 4️⃣ Pie Chart: Delay Reasons
if "Delay Reason" in filtered_df.columns:
    reason_counts = filtered_df["Delay Reason"].value_counts()
    fig4, ax4 = plt.subplots()
    ax4.pie(reason_counts.values, labels=reason_counts.index, autopct='%1.1f%%', startangle=90)
    ax4.set_title("Delay Reasons Distribution")
    st.pyplot(fig4)

# --- Download Section ---
st.markdown("### 📥 Export Filtered Data")
csv = filtered_df.to_csv(index=False).encode("utf-8")
st.download_button("Download CSV", csv, "filtered_data.csv", "text/csv", key='download-csv')

st.success("✅ Dashboard Loaded Successfully!")




