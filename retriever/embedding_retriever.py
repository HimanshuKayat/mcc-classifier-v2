
import pickle

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


class EmbeddingRetriever:

    def __init__(
        self,
        embedding_file="data/mcc_embeddings.pkl",
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    ):

        self.model = SentenceTransformer(model_name)

        with open(embedding_file, "rb") as f:
            data = pickle.load(f)

        self.mcc_profiles = data["profiles"]
        self.embeddings = data["embeddings"]

    def retrieve(self, entity_profile, top_k=20):

        query = self._profile_to_text(entity_profile)

        query_embedding = self.model.encode(
            [query],
            convert_to_numpy=True
        )

        similarities = cosine_similarity(
            query_embedding,
            self.embeddings
        )[0]

        ranked = sorted(
            zip(similarities, self.mcc_profiles),
            key=lambda x: x[0],
            reverse=True
        )

        return [profile for _, profile in ranked[:top_k]]

    def _profile_to_text(self, profile):

        return f"""
Entity Name:
{profile.get("entity_name", "")}

Entity Type:
{profile.get("entity_type", "")}

Summary:
{profile.get("summary", "")}

Primary Business:
{profile.get("primary_business", "")}

Industry:
{profile.get("industry", "")}

Products and Services:
{' '.join(profile.get("products_services", []))}

Target Customers:
{' '.join(profile.get("target_customers", []))}

Business Model:
{profile.get("business_model", "")}

Keywords:
{' '.join(profile.get("keywords", []))}

Aliases:
{' '.join(profile.get("aliases", []))}
"""
