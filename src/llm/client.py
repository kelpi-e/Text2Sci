import os
from google import genai
from google.genai import types

class AIClient:
    def __init__(self):
        self.client = genai.Client(api_key="AIzaSyBeFKnjy5tEvCE-on_FHMggmVZ-Uj0XLAQ")

    def generate(self, user_text: str) -> str:
        full_prompt = user_text
        response = self.client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[types.Part.from_text(text=full_prompt)]
        )
        return response.candidates[0].content.parts[0].text or None
