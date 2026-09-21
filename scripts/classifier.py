from models.llama_model import LlamaModel
from prompts.entity_prompt import EntityPromptBuilder
from prompts.mcc_prompt import MCCPromptBuilder
from retriever.embedding_retriever import EmbeddingRetriever
from utils.wikimedia_metadata import WikimediaMetadata
from parser import JSONParser


class MCCClassifier:

    def __init__(self):

        self.model = LlamaModel()

        self.entity_prompt_builder = EntityPromptBuilder()

        self.mcc_prompt_builder = MCCPromptBuilder()

        self.retriever = EmbeddingRetriever()

        # Load Wikimedia metadata once at startup.
        self.wikimedia_metadata = WikimediaMetadata()

    def classify(self, page_name: str):

        ####################################################
        # STEP 1 : Entity Understanding
        ####################################################

        # Get Wikimedia categories for this article.
        wikimedia_categories = (
            self.wikimedia_metadata.get_categories(page_name)
        )

        print("\n========== WIKIMEDIA METADATA ==========\n")

        if wikimedia_categories:
            print(f"Article: {page_name}")
            print(f"Categories: {wikimedia_categories}")
        else:
            print(
                f"No Wikimedia categories found for: {page_name}"
            )

        print("\n========================================\n")

        entity_prompt = self.entity_prompt_builder.build_prompt(
            page_name,
            wikimedia_categories
        )

        entity_response = self.model.generate(entity_prompt)

        print("\n========== ENTITY UNDERSTANDING ==========\n")
        print(entity_response)
        print("\n==========================================\n")

        entity_profile = JSONParser.parse(entity_response)

        print("\n========== PARSED ENTITY PROFILE ==========\n")
        print(entity_profile)
        print("\n===========================================\n")

        ####################################################
        # STEP 2 : Retrieve Top MCC Candidates
        ####################################################

        candidates = self.retriever.retrieve(
            entity_profile,
            top_k=20
        )

        print("\n========== RETRIEVED MCC CANDIDATES ==========\n")

        for item in candidates:
            print(
                f"{item['mcc']} - {item['industry']}"
            )

        print("\n=============================================\n")

        ####################################################
        # STEP 3 : Final MCC Selection
        ####################################################

        mcc_prompt = self.mcc_prompt_builder.build_prompt(
            entity_profile,
            candidates
        )

        mcc_response = self.model.generate(mcc_prompt)

        print("\n========== FINAL MCC RESPONSE ==========\n")
        print(mcc_response)
        print("\n========================================\n")

        return JSONParser.parse(mcc_response)
