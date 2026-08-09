class PromptBuilder:

    def build_prompt(self, page_name: str):

        prompt = f"""
You are an expert knowledge assistant.

The following input is the title of a Wikipedia article.

Wikipedia Article Title:
{page_name}

Your task is to identify exactly what this article represents.

Do not classify it into any MCC.

Instead, retrieve everything you know about this entity.

Return ONLY valid JSON in the following format.

{{
    "entity_name": "",
    "entity_type": "",
    "summary": "",
    "primary_business": "",
    "industry": "",
    "products_services": [],
    "target_customers": [],
    "business_model": "",
    "parent_company": "",
    "country": "",
    "founded": "",
    "keywords": [],
    "aliases": [],
    "confidence": 0.95
}}
"""

        return prompt
