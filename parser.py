import json
import re


class JSONParser:

    @staticmethod
    def parse(response: str):

        if not response:
            raise ValueError("LLM returned an empty response.")

        response = response.strip()

        response = (
            response
            .replace("```json", "")
            .replace("```", "")
            .strip()
        )

        match = re.search(
            r"\{.*\}",
            response,
            re.DOTALL
        )

        if not match:
            raise ValueError(
                f"No JSON object found.\n\nLLM Response:\n{response}"
            )

        try:
            return json.loads(match.group())
        except json.JSONDecodeError as exc:

            raise ValueError(
                f"Invalid JSON returned by LLM.\n\n"
                f"LLM Response:\n{response}"
            ) from exc

    @staticmethod
    def validate_entity_result(data):

        if "entity_profile" not in data:
            raise ValueError(
                "Entity result missing 'entity_profile'."
            )

        if "commercial_profile" not in data:
            raise ValueError(
                "Entity result missing 'commercial_profile'."
            )

        return data

    @staticmethod
    def validate_predictions(
        data,
        retrieved_candidates
    ):

        if "predictions" not in data:
            raise ValueError(
                "LLM result missing 'predictions'."
            )

        predictions = data["predictions"]

        if not isinstance(predictions, list):
            raise ValueError(
                "'predictions' must be a list."
            )

        if len(predictions) != 5:
            raise ValueError(
                f"Expected exactly 5 predictions, "
                f"received {len(predictions)}."
            )

        retrieved_codes = {
            str(item["mcc"])
            for item in retrieved_candidates
        }

        seen_codes = set()

        for expected_rank, prediction in enumerate(
            predictions,
            start=1
        ):

            if prediction.get("rank") != expected_rank:
                raise ValueError(
                    f"Invalid prediction rank at position "
                    f"{expected_rank}."
                )

            mcc = str(prediction.get("mcc", "")).strip()

            if not mcc:
                raise ValueError(
                    f"Prediction {expected_rank} has no MCC."
                )

            if mcc not in retrieved_codes:
                raise ValueError(
                    f"LLM returned MCC {mcc}, "
                    f"which was not present in Top-20 candidates."
                )

            if mcc in seen_codes:
                raise ValueError(
                    f"Duplicate MCC returned: {mcc}"
                )

            reason = prediction.get("reason", "").strip()

            if not reason:
                raise ValueError(
                    f"Prediction {expected_rank} has no reason."
                )

            seen_codes.add(mcc)

        return predictions
