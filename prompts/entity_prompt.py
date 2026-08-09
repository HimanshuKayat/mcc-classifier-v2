class EntityPromptBuilder:

    def build_prompt(self, page_name: str):

        prompt = f"""
You are an expert knowledge assistant.

The input below is the title of a Wikipedia article.

Wikipedia Article Title:
{page_name}

Your task is to identify exactly what this article represents.

Use your existing knowledge.

Return ONLY valid JSON.

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
    "keywords": [],
    "aliases": []
}}
"""

        return prompt
