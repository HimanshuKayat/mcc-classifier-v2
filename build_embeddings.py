import json
import pickle

from sentence_transformers import SentenceTransformer


MODEL_NAME = (
    "sentence-transformers/all-MiniLM-L6-v2"
)


def profile_to_text(profile):

    return f"""
MCC: {profile.get('mcc', '')}

Industry:
{profile.get('industry', '')}

Category:
{profile.get('category', '')}

Commercial Description:
{profile.get('description', '')}
"""


def main():

    print("Loading embedding model...")

    model = SentenceTransformer(
        MODEL_NAME
    )

    print("Loading MCC profiles...")

    with open(
        "data/mcc_codes.json",
        "r",
        encoding="utf-8"
    ) as f:

        profiles = json.load(f)

    texts = [
        profile_to_text(profile)
        for profile in profiles
    ]

    print(
        f"Generating embeddings for "
        f"{len(texts)} MCC profiles..."
    )

    embeddings = model.encode(
        texts,
        show_progress_bar=True,
        convert_to_numpy=True
    )

    with open(
        "data/mcc_embeddings.pkl",
        "wb"
    ) as f:

        pickle.dump(
            {
                "profiles": profiles,
                "embeddings": embeddings
            },
            f
        )

    print()
    print("Done!")
    print(
        f"Saved {len(profiles)} embeddings "
        f"to data/mcc_embeddings.pkl"
    )


if __name__ == "__main__":
    main()
