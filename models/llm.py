# models/llm.py
# Wraps LLM providers. Keep calls here. This example uses OpenAI's chat completions.
import os
from config.config import LLM_PROVIDER


# Basic OpenAI example
try:
    import openai
except Exception:
    openai = None




def generate_response(prompt: str, system_prompt: str = None, provider: str = None, mode: str = "detailed", max_tokens: int = 512):
    """Generate a response using configured provider. mode in ['concise','detailed']
    Keep it generic so you can add more providers.
    """
    provider = provider or LLM_PROVIDER
    system_prompt = system_prompt or "You are a helpful assistant."
    try:
        if provider == "openai":
            if openai is None:
                raise RuntimeError("openai package not installed")
            # Uses ChatCompletions (gpt-3.5/4 family) as an example
            openai.api_key = os.getenv("OPENAI_API_KEY")
            messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": prompt}]
            # Adjust temperature and max tokens per mode
            if mode == "concise":
                max_tokens = min(200, max_tokens)
                temperature = 0.2
            else:
                temperature = 0.7
            resp = openai.ChatCompletion.create(model="gpt-4o-mini" if False else "gpt-3.5-turbo", messages=messages, max_tokens=max_tokens, temperature=temperature)
            text = resp["choices"][0]["message"]["content"].strip()
            return text
        else:
            # Placeholder for other providers (Groq/Gemini)
            return f"[Simulated {provider} response for mode={mode}] -- Prompt length {len(prompt)}"
    except Exception as e:
        print("LLM generation failed:", e)
        return "Sorry — I couldn't get an LLM response right now."