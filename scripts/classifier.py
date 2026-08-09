from models.llama_model import LlamaModel
from prompts.entity_prompt import EntityPromptBuilder
from prompts.mcc_prompt import MCCPromptBuilder
from retriever.embedding_retriever import EmbeddingRetriever
from scripts.confidence_scorer import ConfidenceScorer
from parser import JSONParser


class MCCClassifier:

    def __init__(
        self,
        debug=False
    ):

        self.debug = debug

        self.model = LlamaModel()

        self.entity_prompt_builder = (
            EntityPromptBuilder()
        )

        self.mcc_prompt_builder = (
            MCCPromptBuilder()
        )

        self.retriever = EmbeddingRetriever()

    def classify(self, page_name: str):

        # ==================================================
        # STEP 1: ENTITY UNDERSTANDING
        # ==================================================

        entity_prompt = (
            self.entity_prompt_builder
            .build_prompt(page_name)
        )

        entity_response = self.model.generate(
            entity_prompt
        )

        entity_result = JSONParser.parse(
            entity_response
        )

        JSONParser.validate_entity_result(
            entity_result
        )

        entity_profile = (
            entity_result["entity_profile"]
        )

        commercial_profile = (
            entity_result["commercial_profile"]
        )

        # ==================================================
        # STEP 2: TOP-20 SEMANTIC RETRIEVAL
        # ==================================================

        candidates = self.retriever.retrieve(
            entity_profile,
            top_k=20
        )

        if len(candidates) != 20:
            raise ValueError(
                f"Expected 20 MCC candidates, "
                f"received {len(candidates)}."
            )

        if self.debug:

            print("\n========== TOP-20 RETRIEVAL ==========\n")

            for item in candidates:

                print(
                    f"{item['mcc']} - "
                    f"{item['industry']} | "
                    f"{item['retrieval_similarity']:.4f}"
                )

        # ==================================================
        # STEP 3: LLM TOP-5 RANKING
        # ==================================================

        mcc_prompt = (
            self.mcc_prompt_builder
            .build_prompt(
                entity_profile,
                candidates
            )
        )

        mcc_response = self.model.generate(
            mcc_prompt
        )

        ranking_result = JSONParser.parse(
            mcc_response
        )

        predictions = (
            JSONParser.validate_predictions(
                ranking_result,
                candidates
            )
        )

        # ==================================================
        # STEP 4: CONFIDENCE
        # ==================================================

        candidate_lookup = {
            str(item["mcc"]): item
            for item in candidates
        }

        final_predictions = []

        for prediction in predictions:

            mcc = str(
                prediction["mcc"]
            )

            selected_mcc = (
                candidate_lookup[mcc]
            )

            retrieval_similarity = float(
                selected_mcc[
                    "retrieval_similarity"
                ]
            )

            semantic_similarity = (
                self.retriever
                .calculate_semantic_similarity(
                    commercial_profile,
                    selected_mcc
                )
            )

            confidence = (
                ConfidenceScorer.calculate(
                    semantic_similarity,
                    retrieval_similarity
                )
            )

            final_predictions.append(
                {
                    "rank": prediction["rank"],
                    "mcc": mcc,
                    "industry": selected_mcc.get(
                        "industry",
                        ""
                    ),
                    "semantic_similarity":
                        semantic_similarity,
                    "retrieval_similarity":
                        retrieval_similarity,
                    "confidence":
                        confidence,
                    "reason":
                        prediction["reason"]
                }
            )

        # ==================================================
        # FINAL RESULT
        # ==================================================

        result = {
            "entity_profile": entity_profile,
            "commercial_profile": commercial_profile,
            "retrieved_candidates": candidates,
            "predictions": final_predictions
        }

        return result
