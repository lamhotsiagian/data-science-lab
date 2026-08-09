import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import pandas as pd
from pipeline import ValidateStage, TransformStage, DataPipeline, IngestStage, ExportStage
import yaml

def test_validate_stage():
    df = pd.DataFrame({'A': [1, None, 3]})
    stage = ValidateStage()
    res = stage.process(df)
    assert len(res) == 2

def test_transform_stage():
    df = pd.DataFrame({'old_col': [1, 2]})
    stage = TransformStage({'old_col': 'new_col'})
    res = stage.process(df)
    assert 'new_col' in res.columns
    
def test_full_pipeline(tmp_path):
    csv_path = tmp_path / "input.csv"
    csv_path.write_text("old_col,B\n1,x\n,y\n2,z")
    
    out_path = tmp_path / "output.parquet"
    
    yaml_path = tmp_path / "config.yaml"
    yaml_content = f"""
    stages:
      - type: ingest
        path: {csv_path}
      - type: validate
      - type: transform
        rename:
          old_col: new_col
      - type: export
        path: {out_path}
    """
    yaml_path.write_text(yaml_content)
    
    pipeline = DataPipeline.from_yaml(str(yaml_path))
    res = pipeline.run()
    
    assert len(res) == 2
    assert 'new_col' in res.columns
    assert out_path.exists()
