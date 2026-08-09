import json


class MCCPromptBuilder:

    def build_prompt(self, entity_profile, candidate_mccs):

        entity_json = json.dumps(
            entity_profile,
            indent=4,
            ensure_ascii=False
        )

        mcc_text = ""

        for index, item in enumerate(candidate_mccs, start=1):

            mcc_text += (
                f"Candidate {index}\n"
                f"MCC: {item.get('mcc', '')}\n"
                f"Industry: {item.get('industry', '')}\n"
                f"Category: {item.get('category', '')}\n"
                f"Description: {item.get('description', '')}\n"
                f"Keywords: {', '.join(item.get('keywords', []))}\n"
                f"Aliases: {', '.join(item.get('aliases', []))}\n\n"
            )

        return f"""
You are an expert Merchant Category Code classifier.

You are given:

1. A structured commercial entity profile.
2. Exactly 20 MCC candidates retrieved through semantic similarity.

Your task is to rank the BEST 5 MCCs from those 20 candidates.

ENTITY PROFILE:

{entity_json}

RETRIEVED MCC CANDIDATES:

{mcc_text}

RULES:

1. Select EXACTLY 5 MCCs.
2. You may ONLY select MCCs appearing in the candidate list.
3. NEVER invent an MCC.
4. NEVER modify an MCC.
5. Do not use any MCC outside the supplied 20 candidates.
6. Rank the candidates from most appropriate to least appropriate.
7. Prefer the MCC that most specifically represents the entity's
   primary commercial activity.
8. Consider the actual product/service, commercial activity, customers,
   delivery method and business model.
9. Do not choose an MCC merely because it shares a word with the entity.
10. Do not assign confidence scores.
11. Provide one concise semantic reason for every selected MCC.
12. Return ONLY valid JSON.
13. Do not include any additional text.

Return exactly:

{{
    "predictions": [
        {{
            "rank": 1,
            "mcc": "",
            "reason": ""
        }},
        {{
            "rank": 2,
            "mcc": "",
            "reason": ""
        }},
        {{
            "rank": 3,
            "mcc": "",
            "reason": ""
        }},
        {{
            "rank": 4,
            "mcc": "",
            "reason": ""
        }},
        {{
            "rank": 5,
            "mcc": "",
            "reason": ""
        }}
    ]
}}
"""
