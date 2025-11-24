# models/llm.py
import requests
import json
import streamlit as st
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

    # Get API key from Streamlit secrets
    PPLX_API_KEY = st.secrets.get("PPLX_API_KEY")
    if not PPLX_API_KEY:
        return "ERROR: PERPLEXITY_API_KEY not found in Streamlit secrets."

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
        "Authorization": f"Bearer {PPLX_API_KEY}"
    }

    try:
        response = requests.post(
            PERPLEXITY_API_URL,
            headers=headers,
            json=payload  # use json=payload instead of data=json.dumps(payload)
        )

        if response.status_code == 401:
            return "ERROR: Invalid Perplexity API key (401 Unauthorized)."

        if response.status_code != 200:
            return f"Perplexity API Error {response.status_code}: {response.text}"

        data = response.json()
        choices = data.get("choices")
        if not choices:
            return "Perplexity API returned no response."

        message = choices[0].get("message", {}).get("content", "")
        return message.strip() if message else "Perplexity API returned empty response."

    except Exception as e:
        print("Perplexity generation failed:", e)
        return "Sorry — I couldn't get a response from Perplexity right now."
