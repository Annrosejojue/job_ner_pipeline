import json
import os
import re
from groq import Groq
from .prompts import VALIDATION_PROMPT

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

class LLMValidator:
    def __init__(self, model="llama-3.1-8b-instant"):
        self.model = model

    def validate(self, text, entities):
        entity_list = [e["text"] for e in entities]

        prompt = f"""
{VALIDATION_PROMPT}

Original Text:
{text}

Predicted Skill Entities:
{entity_list}
"""

        response = client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )

        raw_output = response.choices[0].message.content
        print("\nRAW LLM OUTPUT:\n", raw_output, "\n")

        cleaned = self._extract_json(raw_output)

        try:
            data = json.loads(cleaned)
        except Exception as e:
            print("JSON parsing failed:", e)
            return []

        final = []
        for e in data.get("confirmed", []):
            final.append({"text": e, "label": "SKILL", "source": "bert+llm"})
        for e in data.get("new_entities", []):
            final.append({"text": e, "label": "SKILL", "source": "llm-only"})

        return final

    def _extract_json(self, text):
        # Remove markdown fences
        text = re.sub(r"```.*?```", "", text, flags=re.DOTALL)

        # Extract JSON object
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if match:
            return match.group(0)

        return "{}"
