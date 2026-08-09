import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import pytest
from bi_report import *

def test_kpis():
    df = generate_ecommerce_data(100)
    kpis = compute_kpis(df)
    assert "total_revenue" in kpis
    assert kpis["customer_count"] > 0

def test_trends():
    df = generate_ecommerce_data(100)
    trends = compute_trends(df, "date", "revenue", "ME")
    assert len(trends) > 0

def test_cohort():
    df = generate_ecommerce_data(100)
    cohort = compute_cohort_analysis(df)
    assert cohort.shape[0] > 0
