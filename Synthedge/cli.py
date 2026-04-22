"""
synthedge.cli
=============
Command-line interface.

    synthedge analyze data.csv --target diagnosis
    synthedge fill data.csv --target diagnosis --n-top 3 --out augmented.csv
"""

import argparse
import sys
import pandas as pd
from . import SynthEdge


def cmd_analyze(args):
    df = pd.read_csv(args.file)
    se = SynthEdge(df, target_col=args.target, verbose=True)
    se.analyze()
    print(se.gap_map.to_string(index=False))


def cmd_fill(args):
    df = pd.read_csv(args.file)
    se = SynthEdge(df, target_col=args.target, verbose=True)
    se.analyze()
    aug = se.fill(n_top=args.n_top, ctgan_epochs=args.epochs,
                  use_ctgan=not args.no_ctgan)
    out = args.out or args.file.replace(".csv", "_augmented.csv")
    aug.to_csv(out, index=False)
    print("Saved augmented dataset to: " + out)
    print("Rows: " + str(len(df)) + " -> " + str(len(aug)))


def main():
    parser = argparse.ArgumentParser(
        prog="synthedge",
        description="SynthEdge — diagnosis-first synthetic data augmentation"
    )
    sub = parser.add_subparsers(dest="command")

    # analyze
    p_analyze = sub.add_parser("analyze", help="Detect gap regions in a dataset")
    p_analyze.add_argument("file",   help="Path to CSV file")
    p_analyze.add_argument("--target", default="target", help="Target column name")

    # fill
    p_fill = sub.add_parser("fill", help="Augment dataset by filling gap regions")
    p_fill.add_argument("file",   help="Path to CSV file")
    p_fill.add_argument("--target",   default="target", help="Target column name")
    p_fill.add_argument("--n-top",    type=int, default=3,   help="Top N voxels to synthesize for")
    p_fill.add_argument("--epochs",   type=int, default=100, help="CTGAN training epochs")
    p_fill.add_argument("--no-ctgan", action="store_true",   help="Use Gaussian only (faster)")
    p_fill.add_argument("--out",      default=None,          help="Output CSV path")

    args = parser.parse_args()

    if args.command == "analyze":
        cmd_analyze(args)
    elif args.command == "fill":
        cmd_fill(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
