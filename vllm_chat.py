import requests

def vllm_chat(model, system_prompt, user_prompt, timeout=30):
    headers = {"Content-Type": "application/json"}
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0.7,
        "max_tokens": 400,
    }
    try:
        r = requests.post("http://localhost:8000/v1/chat/completions",
                          headers=headers, json=payload, timeout=timeout)
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"].strip()
    except Exception as e:
        print("❌ vLLM chat error:", e)
        return "(vLLM unavailable)"
