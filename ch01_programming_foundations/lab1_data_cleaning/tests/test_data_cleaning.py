import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import pandas as pd
import pytest
from data_cleaning import *

@pytest.fixture
def messy_df():
    return pd.DataFrame({
        ' Name ': ['Alice ', ' Bob'],
        'Salary': ['$50000', '60,000'],
        'Age': ['25', 'twenty']
    })

def test_clean_column_names(messy_df):
    df = clean_column_names(messy_df)
    assert list(df.columns) == ['name', 'salary', 'age']

def test_clean_numeric(messy_df):
    df = clean_numeric(messy_df, 'Salary')
    assert df['Salary'].iloc[0] == 50000.0
    assert df['Salary'].iloc[1] == 60000.0

def test_pipeline(tmp_path):
    csv_file = tmp_path / "test.csv"
    csv_file.write_text("id,Name,Age,Join Date,Salary,Gender\n1,Alice,25,01/15/2020,$50000,Female\n1,Alice,25,01/15/2020,$50000,Female")
    df = clean_pipeline(str(csv_file))
    assert len(df) == 1
    assert df['salary'].iloc[0] == 50000.0

def test_pipeline_polars(tmp_path):
    csv_file = tmp_path / "test.csv"
    csv_file.write_text("id,Name,Age,Join Date,Salary,Gender\n1,Alice,25,01/15/2020,$50000,Female\n1,Alice,25,01/15/2020,$50000,Female")
    df = clean_pipeline_polars(str(csv_file))
    assert len(df) == 1
    assert df['salary'][0] == 50000
