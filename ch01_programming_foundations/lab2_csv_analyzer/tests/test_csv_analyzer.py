import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import pandas as pd
import subprocess
from csv_analyzer import analyze_csv

def test_analyze_csv():
    df = pd.DataFrame({'A': [1, 2, 3], 'B': ['x', 'y', 'x']})
    stats = analyze_csv(df)
    assert stats['shape'] == (3, 2)
    assert stats['categorical_top5']['B']['x'] == 2

def test_cli(tmp_path):
    csv_file = tmp_path / "test.csv"
    csv_file.write_text("A,B\n1,x\n2,y")
    
    script = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'csv_analyzer.py')
    res = subprocess.run([sys.executable, script, '--file', str(csv_file), '--format', 'json'], capture_output=True, text=True)
    assert res.returncode == 0
    assert '"shape": [' in res.stdout or '"shape":\n' in res.stdout or 'shape' in res.stdout
