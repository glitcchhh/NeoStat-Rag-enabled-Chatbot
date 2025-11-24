# models/llm.py
# Uses Perplexity AI API for chat completions.

import os
import requests
import json
from config.config import LLM_PROVIDER


PERPLEXITY_API_URL = "https://api.perplexity.ai/chat/completions"


def generate_response(
    prompt: str,
    system_prompt: str = None,
    provider: str = None,
    mode: str = "detailed",
    max_tokens: int = 512
):
    """
    Generate a response using Perplexity AI.
    mode: 'concise' or 'detailed'
    """

    provider = provider or LLM_PROVIDER
    system_prompt = system_prompt or "You are a helpful assistant."

    if provider != "perplexity":
        return f"[Simulated {provider} response for mode={mode}]"

    api_key = os.getenv("PERPLEXITY_API_KEY")
    if not api_key:
        return "ERROR: PERPLEXITY_API_KEY not found in environment variables."

    # Mode adjustments
    if mode == "concise":
        max_tokens = min(200, max_tokens)
        temperature = 0.2
    else:
        temperature = 0.7

    payload = {
        "model": "sonar-pro",   
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ],
        "max_tokens": max_tokens,
        "temperature": temperature
    }

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }

    try:
        response = requests.post(
            PERPLEXITY_API_URL,
            headers=headers,
            data=json.dumps(payload)
        )

        if response.status_code != 200:
            return f"Perplexity API Error {response.status_code}: {response.text}"

        data = response.json()
        return data["choices"][0]["message"]["content"].strip()

    except Exception as e:
        print("Perplexity generation failed:", e)
        return "Sorry — I couldn't get a response from Perplexity right now."
