
class ConfidenceScorer:

    SEMANTIC_WEIGHT = 0.80
    RETRIEVAL_WEIGHT = 0.20

    @classmethod
    def calculate(
        cls,
        semantic_similarity,
        retrieval_similarity
    ):

        semantic_similarity = cls._clamp(
            semantic_similarity
        )

        retrieval_similarity = cls._clamp(
            retrieval_similarity
        )

        confidence = (
            cls.SEMANTIC_WEIGHT * semantic_similarity
            +
            cls.RETRIEVAL_WEIGHT * retrieval_similarity
        )

        return float(confidence)

    @staticmethod
    def _clamp(value):

        value = float(value)

        return max(
            0.0,
            min(1.0, value)
        )
