import os

import pandas as pd

from scripts.classifier import MCCClassifier


# ==========================================================
# SETTINGS
# ==========================================================

INPUT_FILE = "data/articles_metadata.xlsx"

OUTPUT_FILE = "outputs/articles_metadata_output.xlsx"

ARTICLE_COLUMN = "Article Name"

SAVE_EVERY = 50

DEBUG = False


# ==========================================================
# ENTITY OUTPUT COLUMNS
# ==========================================================

ENTITY_COLUMNS = [
    "Entity Name",
    "Entity Type",
    "Entity Summary",
    "Primary Business",
    "Industry",
    "Products/Services",
    "Target Customers",
    "Business Model",
    "Parent Company",
    "Country",
    "Keywords",
    "Aliases",
]


# ==========================================================
# MCC OUTPUT COLUMNS
# ==========================================================

MCC_COLUMNS = []

for rank in range(1, 6):

    MCC_COLUMNS.extend(
        [
            f"MCC Top {rank}",
            f"MCC Top {rank} Industry",
            f"MCC Top {rank} Semantic Similarity",
            f"MCC Top {rank} Retrieval Similarity",
            f"MCC Top {rank} Confidence",
            f"MCC Top {rank} Reason",
        ]
    )


STATUS_COLUMNS = [
    "Processing Status",
    "Processing Error",
]


# ==========================================================
# HELPERS
# ==========================================================

def ensure_output_columns(df):

    for column in (
        ENTITY_COLUMNS
        + MCC_COLUMNS
        + STATUS_COLUMNS
    ):

        if column not in df.columns:
            df[column] = ""

    return df


def safe_string_list(value):

    if not value:
        return ""

    return "; ".join(
        str(item)
        for item in value
    )


def print_entity_profile(entity):

    print("\nEntity Profile")
    print("-" * 60)

    print(
        f"Entity Name: "
        f"{entity.get('entity_name', '')}"
    )

    print(
        f"Entity Type: "
        f"{entity.get('entity_type', '')}"
    )

    print(
        f"Summary: "
        f"{entity.get('summary', '')}"
    )

    print(
        f"Primary Business: "
        f"{entity.get('primary_business', '')}"
    )

    print(
        f"Industry: "
        f"{entity.get('industry', '')}"
    )

    print(
        f"Products/Services: "
        f"{safe_string_list(entity.get('products_services'))}"
    )

    print(
        f"Target Customers: "
        f"{safe_string_list(entity.get('target_customers'))}"
    )

    print(
        f"Business Model: "
        f"{entity.get('business_model', '')}"
    )

    print(
        f"Parent Company: "
        f"{entity.get('parent_company', '')}"
    )

    print(
        f"Country: "
        f"{entity.get('country', '')}"
    )

    print(
        f"Keywords: "
        f"{safe_string_list(entity.get('keywords'))}"
    )

    print(
        f"Aliases: "
        f"{safe_string_list(entity.get('aliases'))}"
    )


def print_predictions(predictions):

    print("\nTop-5 MCC Predictions")
    print("-" * 60)

    for prediction in predictions:

        confidence = (
            prediction["confidence"] * 100
        )

        print(
            f"\n{prediction['rank']}. "
            f"{prediction['mcc']} - "
            f"{prediction['industry']}"
        )

        print(
            f"   Confidence: "
            f"{confidence:.2f}%"
        )

        print(
            f"   Reason: "
            f"{prediction['reason']}"
        )


def write_entity_to_row(
    df,
    index,
    entity
):

    df.at[index, "Entity Name"] = (
        entity.get("entity_name", "")
    )

    df.at[index, "Entity Type"] = (
        entity.get("entity_type", "")
    )

    df.at[index, "Entity Summary"] = (
        entity.get("summary", "")
    )

    df.at[index, "Primary Business"] = (
        entity.get("primary_business", "")
    )

    df.at[index, "Industry"] = (
        entity.get("industry", "")
    )

    df.at[index, "Products/Services"] = (
        safe_string_list(
            entity.get("products_services")
        )
    )

    df.at[index, "Target Customers"] = (
        safe_string_list(
            entity.get("target_customers")
        )
    )

    df.at[index, "Business Model"] = (
        entity.get("business_model", "")
    )

    df.at[index, "Parent Company"] = (
        entity.get("parent_company", "")
    )

    df.at[index, "Country"] = (
        entity.get("country", "")
    )

    df.at[index, "Keywords"] = (
        safe_string_list(
            entity.get("keywords")
        )
    )

    df.at[index, "Aliases"] = (
        safe_string_list(
            entity.get("aliases")
        )
    )


def write_predictions_to_row(
    df,
    index,
    predictions
):

    for prediction in predictions:

        rank = prediction["rank"]

        df.at[
            index,
            f"MCC Top {rank}"
        ] = prediction["mcc"]

        df.at[
            index,
            f"MCC Top {rank} Industry"
        ] = prediction["industry"]

        df.at[
            index,
            f"MCC Top {rank} Semantic Similarity"
        ] = prediction[
            "semantic_similarity"
        ]

        df.at[
            index,
            f"MCC Top {rank} Retrieval Similarity"
        ] = prediction[
            "retrieval_similarity"
        ]

        df.at[
            index,
            f"MCC Top {rank} Confidence"
        ] = prediction[
            "confidence"
        ]

        df.at[
            index,
            f"MCC Top {rank} Reason"
        ] = prediction["reason"]


def save_output(df):

    output_directory = os.path.dirname(
        OUTPUT_FILE
    )

    if output_directory:
        os.makedirs(
            output_directory,
            exist_ok=True
        )

    df.to_excel(
        OUTPUT_FILE,
        index=False
    )


# ==========================================================
# MAIN
# ==========================================================

def main():

    print("=" * 70)
    print("SEMANTIC MCC CLASSIFIER")
    print("=" * 70)

    if not os.path.exists(INPUT_FILE):
        raise FileNotFoundError(
            f"Input file not found: {INPUT_FILE}"
        )

    # ------------------------------------------------------
    # LOAD INPUT / RESUME OUTPUT
    # ------------------------------------------------------

    if os.path.exists(OUTPUT_FILE):

        print(
            f"Existing output found. "
            f"Loading for resume: {OUTPUT_FILE}"
        )

        df = pd.read_excel(
            OUTPUT_FILE
        )

    else:

        print(
            f"Loading input: {INPUT_FILE}"
        )

        df = pd.read_excel(
            INPUT_FILE
        )

    if ARTICLE_COLUMN not in df.columns:

        raise ValueError(
            f"Required column "
            f"'{ARTICLE_COLUMN}' "
            f"was not found in Excel."
        )

    df = ensure_output_columns(df)

    classifier = MCCClassifier(
        debug=DEBUG
    )

    total = len(df)

    processed_since_save = 0

    # ------------------------------------------------------
    # PROCESS EACH ARTICLE
    # ------------------------------------------------------

    for index, row in df.iterrows():

        status = str(
            row.get(
                "Processing Status",
                ""
            )
        ).strip()

        # Resume support
        if status == "Completed":
            continue

        article_name = row[
            ARTICLE_COLUMN
        ]

        if pd.isna(article_name):

            df.at[
                index,
                "Processing Status"
            ] = "Failed"

            df.at[
                index,
                "Processing Error"
            ] = "Empty Article Name"

            processed_since_save += 1

            continue

        article_name = str(
            article_name
        ).strip()

        if not article_name:

            df.at[
                index,
                "Processing Status"
            ] = "Failed"

            df.at[
                index,
                "Processing Error"
            ] = "Empty Article Name"

            processed_since_save += 1

            continue

        print("\n" + "=" * 70)
        print(
            f"ROW {index + 1}/{total}"
        )
        print(
            f"ARTICLE: {article_name}"
        )
        print("=" * 70)

        try:

            result = classifier.classify(
                article_name
            )

            entity = result[
                "entity_profile"
            ]

            predictions = result[
                "predictions"
            ]

            # Write entity profile
            write_entity_to_row(
                df,
                index,
                entity
            )

            # Write Top-5
            write_predictions_to_row(
                df,
                index,
                predictions
            )

            df.at[
                index,
                "Processing Status"
            ] = "Completed"

            df.at[
                index,
                "Processing Error"
            ] = ""

            # Console
            print_entity_profile(
                entity
            )

            print_predictions(
                predictions
            )

        except Exception as exc:

            df.at[
                index,
                "Processing Status"
            ] = "Failed"

            df.at[
                index,
                "Processing Error"
            ] = str(exc)

            print(
                f"\nERROR: {exc}"
            )

        processed_since_save += 1

        # --------------------------------------------------
        # SAVE EVERY 50 PROCESSED ROWS
        # --------------------------------------------------

        if (
            processed_since_save >= SAVE_EVERY
        ):

            save_output(df)

            print(
                f"\n>>> Progress saved "
                f"after {SAVE_EVERY} processed rows."
            )

            processed_since_save = 0

    # ------------------------------------------------------
    # FINAL SAVE
    # ------------------------------------------------------

    save_output(df)

    print("\n" + "=" * 70)
    print("PROCESSING COMPLETE")
    print(
        f"Output saved to: {OUTPUT_FILE}"
    )
    print("=" * 70)


if __name__ == "__main__":
    main()
