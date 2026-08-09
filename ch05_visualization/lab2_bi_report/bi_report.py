import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

def generate_ecommerce_data(n_orders):
    np.random.seed(42)
    dates = pd.date_range("2023-01-01", periods=n_orders, freq="D")
    customers = np.random.randint(1, n_orders//3, n_orders)
    categories = np.random.choice(["A", "B", "C"], n_orders)
    revenues = np.random.uniform(10, 100, n_orders)
    quantities = np.random.randint(1, 5, n_orders)
    regions = np.random.choice(["N", "S", "E", "W"], n_orders)
    return pd.DataFrame({"order_id": range(n_orders), "date": dates, "customer_id": customers, 
                         "product_category": categories, "revenue": revenues, "quantity": quantities, "region": regions})

def compute_kpis(df):
    return {
        "total_revenue": df["revenue"].sum(),
        "avg_order_value": df["revenue"].mean(),
        "customer_count": df["customer_id"].nunique(),
        "orders_per_customer": len(df) / df["customer_id"].nunique()
    }

def compute_trends(df, date_col, metric_col, freq="ME"):
    return df.set_index(date_col).resample(freq)[metric_col].sum().reset_index()

def compute_cohort_analysis(df):
    df = df.copy()
    df["order_month"] = df["date"].dt.to_period("M")
    df["cohort"] = df.groupby("customer_id")["date"].transform("min").dt.to_period("M")
    cohort_data = df.groupby(["cohort", "order_month"])["customer_id"].nunique().reset_index()
    return cohort_data.pivot(index="cohort", columns="order_month", values="customer_id")

def compute_category_breakdown(df):
    return df.groupby("product_category")["revenue"].sum()

def compute_regional_performance(df):
    return df.groupby("region")["revenue"].sum()

def create_kpi_cards(kpis):
    fig, ax = plt.subplots()
    ax.text(0.5, 0.5, str(kpis))
    return fig

def plot_trend(trend_df):
    fig, ax = plt.subplots()
    ax.plot(trend_df.iloc[:,0], trend_df.iloc[:,1])
    return fig

def plot_cohort_heatmap(cohort_df):
    fig, ax = plt.subplots()
    ax.imshow(cohort_df.fillna(0).values)
    return fig

def plot_category_pie(breakdown):
    fig, ax = plt.subplots()
    ax.pie(breakdown.values, labels=breakdown.index)
    return fig
