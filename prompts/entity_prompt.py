class EntityPromptBuilder:

    def build_prompt(
        self,
        page_name: str,
        wikimedia_categories: str = ""
    ):

        if wikimedia_categories:
            wikimedia_section = f"""
Wikimedia Categories:
{wikimedia_categories}
"""
        else:
            wikimedia_section = """
Wikimedia Categories:
No Wikimedia category metadata was found for this article.
"""

        prompt = f"""
You are an expert knowledge assistant.

The input below is the title of a Wikipedia article.

Wikipedia Article Title:
{page_name}

{wikimedia_section}

The Wikimedia categories are additional contextual information
about the article. Use them to better understand what the entity
represents, especially when the article title is a niche, ambiguous,
or proper-noun entity.

Use your existing knowledge together with the Wikimedia categories.

Do not assume that every Wikimedia category directly describes the
entity's primary business. Use the categories as supporting context.

Your task is to identify exactly what this article represents.

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
