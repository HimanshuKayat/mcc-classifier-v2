import pandas as pd
from pathlib import Path
import argparse


def parquet_to_excel(input_file, output_file=None):
    input_path = Path(input_file)

    if not input_path.exists():
        raise FileNotFoundError(
            f"Parquet file not found: {input_path.resolve()}"
        )

    if input_path.suffix.lower() != ".parquet":
        raise ValueError("Input file must have a .parquet extension")

    print(f"Reading: {input_path.resolve()}")

    df = pd.read_parquet(input_path)

    print(f"Rows: {len(df):,}")
    print(f"Columns: {len(df.columns)}")

    # If no output is specified, create Excel
    # in the SAME directory as the Parquet file.
    if output_file is None:
        output_path = input_path.with_suffix(".xlsx")
    else:
        output_path = Path(output_file)

        # Create parent directory if necessary
        output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Writing: {output_path.resolve()}")

    df.to_excel(
        output_path,
        index=False,
        engine="openpyxl"
    )

    print("\nConversion completed successfully.")
    print(f"Output: {output_path.resolve()}")
    print(f"Rows: {len(df):,}")
    print(f"Columns: {len(df.columns)}")

    print("\nColumns:")
    for i, column in enumerate(df.columns, start=1):
        print(f"{i}. {column}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Convert a Parquet file to Excel"
    )

    parser.add_argument(
        "input",
        help="Path to the input Parquet file"
    )

    parser.add_argument(
        "-o",
        "--output",
        default=None,
        help=(
            "Optional output Excel path. "
            "If omitted, Excel is created beside the Parquet file."
        )
    )

    args = parser.parse_args()

    parquet_to_excel(
        args.input,
        args.output
    )
