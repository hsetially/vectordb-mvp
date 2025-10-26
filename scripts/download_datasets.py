import argparse
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from benchmarks.datasets import download_dataset

def main():
    parser = argparse.ArgumentParser(description="Download standard ANN benchmark datasets")
    parser.add_argument(
        '--dataset', 
        type=str, 
        required=True, 
        choices=['sift', 'glove', 'mnist'],
        help='Name of the dataset to download.'
    )
    parser.add_argument(
        '--output', 
        type=str, 
        default='./data',
        help='Directory to save the dataset file.'
    )
    
    args = parser.parse_args()
    
    try:
        download_dataset(name=args.dataset, data_dir=args.output)
    except Exception as e:
        print(f"An error occurred: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
