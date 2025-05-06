from typing import List, Dict
from openai import OpenAI

class OpenAIProvider:
    def __init__(self, model: str = "gpt-4o-mini"):
        self.model = model
        self.client = OpenAI()

    def chat(self, messages: List[Dict[str, str]], temperature: float = .2) -> str:
        resp = self.client.chat.completions.create(
            model=self.model, messages=messages, temperature=temperature
        )
        return resp.choices[0].message.content