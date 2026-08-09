import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import preprocessing_pipeline
import pandas as pd
import numpy as np

def test_pipeline_fit_transform():
    df = pd.DataFrame({
        'age': [25, np.nan, 30],
        'income': [50000, 60000, 70000],
        'city': ['A', 'B', np.nan],
        'signup_date': ['2023-01-01', '2023-02-01', '2023-03-01']
    })
    df['signup_date'] = pd.to_datetime(df['signup_date'])
    
    config = preprocessing_pipeline.create_default_config(df)
    transformed, pipeline = preprocessing_pipeline.fit_transform_pipeline(df, config)
    
    assert transformed.shape[0] == 3
    assert not pd.isna(transformed).any()

def test_config_loading(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text("numeric_cols: ['a']\ncategorical_cols: ['b']")
    
    config = preprocessing_pipeline.PreprocessingConfig.from_yaml(config_path)
    assert config.numeric_cols == ['a']
    assert config.categorical_cols == ['b']
