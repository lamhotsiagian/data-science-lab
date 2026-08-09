import argparse
import pandas as pd
import json
import logging
import sys

def setup_logging(level):
    logging.basicConfig(level=getattr(logging, level.upper()), format='%(levelname)s:%(message)s')

def analyze_csv(df):
    stats = {
        'shape': df.shape,
        'dtypes': df.dtypes.astype(str).to_dict(),
        'null_counts': df.isnull().sum().to_dict(),
        'unique_counts': df.nunique().to_dict(),
        'numeric_stats': df.describe().to_dict(),
        'categorical_top5': {}
    }
    for col in df.select_dtypes(include=['object', 'category']).columns:
        stats['categorical_top5'][col] = df[col].value_counts().head(5).to_dict()
    return stats

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', required=True)
    parser.add_argument('--output', default=None)
    parser.add_argument('--format', choices=['text', 'json'], default='text')
    parser.add_argument('--log-level', default='INFO')
    args = parser.parse_args()

    setup_logging(args.log_level)
    logging.info(f"Analyzing {args.file}")
    
    try:
        df = pd.read_csv(args.file)
    except Exception as e:
        logging.error(f"Error reading {args.file}: {e}")
        sys.exit(1)

    stats = analyze_csv(df)
    
    if args.format == 'json':
        out = json.dumps(stats, indent=2)
    else:
        out = f"Shape: {stats['shape']}\nNulls: {stats['null_counts']}"
        
    if args.output:
        with open(args.output, 'w') as f:
            f.write(out)
    else:
        print(out)

if __name__ == "__main__":
    main()
