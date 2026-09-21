import pandas as pd
from pathlib import Path


class WikimediaMetadata:

    def __init__(
        self,
        metadata_file="data/wikimedia/india_articles_metadata_v3.xlsx"
    ):
        self.metadata_file = Path(metadata_file)

        if not self.metadata_file.exists():
            raise FileNotFoundError(
                f"Wikimedia metadata file not found: "
                f"{self.metadata_file.resolve()}"
            )

        df = pd.read_excel(self.metadata_file)

        required_columns = ["article", "categories"]

        missing = [
            column
            for column in required_columns
            if column not in df.columns
        ]

        if missing:
            raise ValueError(
                f"Missing required Wikimedia columns: {missing}"
            )

        self.metadata = {}

        for _, row in df.iterrows():

            article = row["article"]

            if pd.isna(article):
                continue

            article = str(article).strip()

            if not article:
                continue

            categories = row["categories"]

            if pd.isna(categories):
                categories = ""

            categories = str(categories).strip()

            key = self._normalize(article)

            # If an article occurs more than once,
            # combine its categories rather than overwriting them.
            if key in self.metadata:

                existing = self.metadata[key]

                if categories and categories not in existing:
                    self.metadata[key] = (
                        existing + " | " + categories
                    )

            else:
                self.metadata[key] = categories

        print(
            f"Loaded {len(self.metadata):,} Wikimedia article "
            f"metadata records."
        )

    @staticmethod
    def _normalize(text):

        return " ".join(
            str(text).strip().lower().split()
        )

    def get_categories(self, article):

        if not article:
            return ""

        key = self._normalize(article)

        return self.metadata.get(key, "")
