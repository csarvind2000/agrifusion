# 🌾 Migration Guide: Converting AgriFusion Multi-Agent System from **Ollama** to **vLLM**

This guide explains how to migrate your **AgriFusion Multi-Agentic Smart Farming Assistant** from **Ollama** (local lightweight inference) to **vLLM** (OpenAI-compatible high-performance inference server).

---

## 🧭 1. Overview

### **Goal**

Switch the AgriFusion system’s backend from **Ollama (`:11434`)** to **vLLM (`:8000`)** while keeping the **multi-agent reasoning**, **multilingual Gradio UI**, and **tabular + vision pipelines** intact.

### **Key Differences**

| Feature / Aspect    | Ollama                                               | vLLM                                                     |
| ------------------- | ---------------------------------------------------- | -------------------------------------------------------- |
| **API type**        | Proprietary REST API (`/api/chat`)                   | OpenAI-compatible REST API (`/v1/chat/completions`)      |
| **Default host**    | `http://127.0.0.1:11434`                             | `http://127.0.0.1:8000`                                  |
| **Model reference** | e.g., `llama3:latest`                                | e.g., `meta-llama/Meta-Llama-3-8B-Instruct`              |
| **Response field**  | `message.content`                                    | `choices[0].message.content`                             |
| **Compute style**   | Optimized local quantized inference (CPU / low-VRAM) | High-throughput GPU batching & tensor-parallel inference |
| **Ease of use**     | Turn-key binary                                      | Requires Python environment setup                        |

---

## ⚙️ 2. Installation Requirements

```bash
# 1️⃣ (Optional) new environment
conda create -n agrifusion_vllm python=3.10 -y
conda activate agrifusion_vllm

# 2️⃣ PyTorch (per your CUDA / MPS)
pip install torch torchvision torchaudio

# 3️⃣ vLLM inference engine
pip install vllm

# 4️⃣ Optional OpenAI SDK client
pip install openai

# 5️⃣ Project dependencies
pip install gradio timm pillow requests joblib numpy pandas whisper gtts
```

---

## 🚀 3. Launch the vLLM Server

```bash
python -m vllm.entrypoints.openai.api_server \
  --model meta-llama/Meta-Llama-3-8B-Instruct \
  --port 8000 \
  --max-num-seqs 8 \
  --gpu-memory-utilization 0.9
```

Verify via **[http://localhost:8000/docs](http://localhost:8000/docs)**.
API endpoint → **[http://localhost:8000/v1/chat/completions](http://localhost:8000/v1/chat/completions)**

---

## 🧩 4. Replace Ollama Calls

### **Old (Ollama)**

```python
payload = {
  "model": "llama3:latest",
  "messages": [
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": user_prompt},
  ],
}
r = requests.post("http://127.0.0.1:11434/api/chat", json=payload, timeout=20)
txt = (r.json().get("message", {}) or {}).get("content", "")
```

### **New (vLLM) → helper `vllm_chat.py`**

```python
import requests
VLLM_API_URL = "http://127.0.0.1:8000/v1/chat/completions"

def vllm_chat(model, system_prompt, user_prompt, timeout=30):
    headers = {"Content-Type": "application/json"}
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "max_tokens": 512,
        "temperature": 0.7,
    }
    try:
        r = requests.post(VLLM_API_URL, headers=headers, json=payload, timeout=timeout)
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"].strip()
    except Exception as e:
        print("❌ vLLM chat error:", e)
        return "(vLLM unavailable)"
```

---

## 🧠 5. Integrate into Your Scripts

### In `crop_multimodal_app_multilang.py`

```python
from vllm_chat import vllm_chat
def llm_chat(model, system, user, timeout=30):
    return vllm_chat(model, system, user, timeout)
```

Replace all `ollama_chat(...)` → `llm_chat(...)`.

### In `agri_fusion_multiagent.py`

```python
from vllm_chat import vllm_chat
txt = vllm_chat(VLLM_MODEL, system, query, timeout=LLM_TIMEOUT)
```

---

## 🧾 6. Configuration Variables

```python
VLLM_MODEL = "meta-llama/Meta-Llama-3-8B-Instruct"
VLLM_PORT = 8000
VLLM_BASE_URL = f"http://127.0.0.1:{VLLM_PORT}/v1"
```

---

## 🧪 7. Test the Integration

```bash
# 1️⃣ Start vLLM
python -m vllm.entrypoints.openai.api_server --model meta-llama/Meta-Llama-3-8B-Instruct

# 2️⃣ Launch Gradio
python agri_fusion_multiagent_ui_vllm.py
```

Access via **[http://127.0.0.1:7863](http://127.0.0.1:7863)**

---

## 🔧 8. Optional — OpenAI Python Client

```python
from openai import OpenAI
client = OpenAI(base_url="http://127.0.0.1:8000/v1", api_key="EMPTY")

def vllm_chat(model, system_prompt, user_prompt, temperature=0.7):
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=temperature,
        max_tokens=512,
    )
    return resp.choices[0].message.content.strip()
```

---

## ⚖️ 9. Why LLaMA 8B Works with **Ollama** but not with **vLLM** (by default)

| Factor                  | Ollama                                                                                    | vLLM                                                       |
| ----------------------- | ----------------------------------------------------------------------------------------- | ---------------------------------------------------------- |
| **Model packaging**     | Ships **pre-quantized** models (e.g., Q4_K_M) optimized for **CPU / low-VRAM** execution. | Expects **full-precision** HF models (FP16 / BF16 / FP32). |
| **Hardware dependency** | Runs on laptops (Apple M-series / AVX2 CPUs) using gguf quantization.                     | Needs GPUs (≥ 16 GB VRAM for 8B).                          |
| **Memory handling**     | Streams layers directly from disk (lazy load).                                            | Fully loads weights into GPU memory.                       |
| **Backend runtime**     | Custom C++ + Metal / llama.cpp.                                                           | CUDA / Triton kernel graph runtime.                        |

### 🧠 Tricks to Run Large Models (like 8B) in vLLM

1. **Quantized weights**: convert HF weights to `GPTQ`, `AWQ`, or `bitsandbytes` 4-bit and point vLLM to that folder (note: partial support).
2. **Reduce `max-model-len`**: cuts KV-cache size and memory use.
3. **Multi-GPU**: launch vLLM with tensor-parallel shards, e.g.
   `--tensor-parallel-size 2` for dual GPU.
4. **CPU offloading** *(experimental)*: use `--swap-space 16` GB if VRAM < 16 GB.
5. **Use smaller model** for dev: `microsoft/phi-1_5` or `Mistral-7B-Instruct`.

If your laptop lacks ≥ 16 GB VRAM, Ollama’s GGUF quantized LLaMA 8B remains the most practical local option.

---

## ✅ 10. Migration Checklist

| Step | Action                        | Done |
| ---- | ----------------------------- | ---- |
| 1    | Install vLLM + deps           | ☐    |
| 2    | Launch vLLM server            | ☐    |
| 3    | Add `vllm_chat.py` helper     | ☐    |
| 4    | Replace `ollama_chat()` calls | ☐    |
| 5    | Update model & base URL       | ☐    |
| 6    | Test end-to-end Gradio app    | ☐    |

---

## 📚 References

* [vLLM Documentation](https://docs.vllm.ai)
* [OpenAI-Compatible API Schema](https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html)
* [Meta LLaMA 3 on Hugging Face](https://huggingface.co/meta-llama)
* [Ollama Quantization Docs](https://github.com/ollama/ollama)

---

**Document:** *Migration Notes — AgriFusion (Ollama → vLLM)*
**Author:** Arvind C. S.
**Date:** November 2025
