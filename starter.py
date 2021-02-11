import argparse
import pandas as pd
from deepan.create_table import DEEPAN

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('expression', '--file', help="Path to expression profile")
    parser.add_argument('-o', '--output', type=str, required=True, help="Output Path")

    args = parser.parse_args()
    inputfile = pd.read_csv(args.expression, sep='\t', index_col=0)
    #tar: df = pd.read_csv('sample.tar.gz', compression='gzip', header=0, sep=' ', quotechar='"', error_bad_lines=False)
    result = DEEPAN(inputfile)
    result.to_csv(args.output, sep='\t')