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

        results = []

        for similarity, profile in ranked[:top_k]:

            result = dict(profile)

            result["retrieval_similarity"] = float(similarity)

            results.append(result)

        return results

    def calculate_semantic_similarity(
        self,
        commercial_profile,
        mcc_profile
    ):
        """
        Compare the independent commercial profile against
        the selected MCC profile.

        This is the 80% component of confidence.
        """

        commercial_text = self._commercial_profile_to_text(
            commercial_profile
        )

        mcc_text = self._mcc_profile_to_text(
            mcc_profile
        )

        embeddings = self.model.encode(
            [
                commercial_text,
                mcc_text
            ],
            convert_to_numpy=True
        )

        similarity = cosine_similarity(
            [embeddings[0]],
            [embeddings[1]]
        )[0][0]

        return float(similarity)

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

Country:
{profile.get("country", "")}
"""

    def _commercial_profile_to_text(self, profile):

        return f"""
Commercial Activity:
{profile.get("commercial_activity", "")}

Primary Offering:
{profile.get("primary_offering", "")}

Delivery Method:
{profile.get("delivery_method", "")}

Customer Type:
{profile.get("customer_type", "")}

Revenue Model:
{profile.get("revenue_model", "")}

Business Context:
{profile.get("business_context", "")}
"""

    def _mcc_profile_to_text(self, profile):

        return f"""
MCC:
{profile.get("mcc", "")}

Industry:
{profile.get("industry", "")}

Category:
{profile.get("category", "")}

Description:
{profile.get("description", "")}

Keywords:
{' '.join(profile.get("keywords", []))}

Aliases:
{' '.join(profile.get("aliases", []))}
"""
