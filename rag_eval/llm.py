import os
import time
from groq import Groq

import config

_client = None


def _get_client():
    global _client
    if _client is None:
        _client = Groq(api_key=os.environ["GROQ_API_KEY"])
    return _client


def call(prompt, temperature=0.0, retries=3, wait=3):
    prompt = prompt.encode("utf-8", errors="replace").decode("utf-8")
    for attempt in range(retries):
        try:
            resp = _get_client().chat.completions.create(
                model=config.LLM_MODEL,
                temperature=temperature,
                messages=[{"role": "user", "content": prompt}],
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(wait)
                continue
            print(f"  llm error after {retries} tries: {e}")
            return ""
    return ""
