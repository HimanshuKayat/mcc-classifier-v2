import json
import re


class JSONParser:

    @staticmethod
    def parse(response: str):

        if not response:
            raise ValueError("LLM returned an empty response.")

        response = response.strip()

        response = response.replace("```json", "").replace("```", "").strip()

        match = re.search(r"\{.*\}", response, re.DOTALL)

        if not match:
            raise ValueError(f"No JSON object found.\n\nLLM Response:\n{response}")

        return json.loads(match.group())
