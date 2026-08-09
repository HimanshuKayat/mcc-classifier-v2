import json


class MCCPromptBuilder:

    def build_prompt(self, entity_profile, candidate_mccs):

        entity_json = json.dumps(entity_profile, indent=4)

        mcc_text = ""

        for item in candidate_mccs:

            mcc_text += (
                f"MCC: {item.get('mcc','')}\n"
                f"Industry: {item.get('industry','')}\n"
                f"Category: {item.get('category','')}\n"
                f"Description: {item.get('description','')}\n"
                f"Keywords: {', '.join(item.get('keywords', []))}\n"
                f"Aliases: {', '.join(item.get('aliases', []))}\n\n"
            )

        prompt = f"""
You are an expert Merchant Category Code (MCC) classifier.

Below is a structured business profile.

Business Profile

{entity_json}

Below are the MOST SEMANTICALLY SIMILAR MCC profiles retrieved from the database.

{mcc_text}

Your task is to compare the Business Profile with ONLY these candidate MCC profiles.

Use all information available in the Business Profile.

Do NOT compare against any MCC outside this list.

Think internally.

Do NOT explain your reasoning.

Do NOT list candidate MCCs.

Do NOT rank MCCs.

Return ONLY one valid JSON object.

{{
    "mcc": "0000",
    "industry": "Industry Name",
    "confidence": 0.95,
    "reason": "One concise sentence explaining the selected MCC."
}}
"""

        return prompt
