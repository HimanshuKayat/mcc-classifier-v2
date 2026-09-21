import pandas as pd
from pathlib import Path
import argparse


def parquet_to_excel(input_file, output_file=None):
    input_path = Path(input_file)

    if not input_path.exists():
        raise FileNotFoundError(f"Parquet file not found: {input_path}")

    if input_path.suffix.lower() != ".parquet":
        raise ValueError("Input file must have a .parquet extension")

    # Read Parquet
    print(f"Reading: {input_path}")
    df = pd.read_parquet(input_path)

    print(f"Rows: {len(df):,}")
    print(f"Columns: {len(df.columns)}")

    # Default output name
    if output_file is None:
        output_file = input_path.with_suffix(".xlsx")

    output_path = Path(output_file)

    # Convert to Excel
    print(f"Writing: {output_path}")
    df.to_excel(output_path, index=False, engine="openpyxl")

    print("\nConversion completed successfully.")
    print(f"Output: {output_path}")
    print(f"Rows: {len(df):,}")
    print(f"Columns: {len(df.columns)}")

    # Display column names so we can inspect the dataset
    print("\nColumns:")
    for i, column in enumerate(df.columns, start=1):
        print(f"{i}. {column}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert a Parquet file to Excel (.xlsx)"
    )

    parser.add_argument(
        "input",
        help="Path to the input Parquet file"
    )

    parser.add_argument(
        "-o",
        "--output",
        help="Path for the output Excel file",
        default=None
    )

    args = parser.parse_args()

    parquet_to_excel(args.input, args.output)
