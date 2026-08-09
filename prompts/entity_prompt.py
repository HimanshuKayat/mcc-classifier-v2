class EntityPromptBuilder:

    def build_prompt(self, page_name: str):

        return f"""
You are an expert commercial entity understanding model.

The input is the title of a Wikipedia article.

Wikipedia Article Title:
{page_name}

Your task is to understand the entity represented by this article.

Do NOT classify the entity into an MCC.
Do NOT mention or predict any MCC.
Do NOT use an MCC database.

Extract commercially relevant information about the entity.

The entity profile must describe what the entity actually is, what it
primarily does, what it sells or provides, who its customers are, how
its business operates, and how its products/services are delivered.

Also create an independent commercial profile. This profile will later
be compared semantically against MCC profiles. It MUST be created without
seeing any MCC candidates.

Return ONLY valid JSON.

Use exactly this structure:

{{
    "entity_profile": {{
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
    }},

    "commercial_profile": {{
        "commercial_activity": "",
        "primary_offering": "",
        "delivery_method": "",
        "customer_type": "",
        "revenue_model": "",
        "business_context": ""
    }}
}}

Important rules:

1. Do not invent facts unnecessarily.
2. Prefer commercially meaningful information over trivia.
3. The primary_business must describe the entity's main commercial activity.
4. products_services must contain actual products or services.
5. target_customers must describe the primary customer group.
6. business_model must describe how the entity generates revenue.
7. commercial_profile must independently summarize the commercial nature
   of the entity.
8. Do not mention MCCs anywhere in the response.
9. Do not include confidence scores.
"""
