#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AgriFusion Multi-Agentic Smart Farming Assistant (vLLM Edition – Phi-1.5)
================================================================
LLM: vLLM OpenAI-compatible API (microsoft/phi-1_5)
Frontend: Gradio
Agents: Router, Vision, Voice, Tabular, Text, Reasoning, TTS
"""

import os, json, subprocess, socket, time
from pathlib import Path
from autogen import AssistantAgent, GroupChat, GroupChatManager
from dotenv import load_dotenv
from vllm_chat import vllm_chat
from crop_multimodal_app_multilang_vllm import (
    LANGS, UI, ui_t, build_system_prompt,
    load_tabular_artifacts, predict_tabular,
    load_vision_model, predict_image, whisper_transcribe,
    _tabular_is_active, _tts_if_possible, build_interface
)

# ============================================================
# 🔧 Environment & Config
# ============================================================
load_dotenv()

VLLM_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
MODEL_DIR = "./models/tinyllama-1.1b-chat"
DEFAULT_LANG = "en"
WHISPER_SIZE = "base"
LLM_TIMEOUT = 20
VLLM_PORT = 8000
VLLM_BASE_URL = f"http://127.0.0.1:{VLLM_PORT}/v1"

# ============================================================
# 🧠 Utility Functions — Model Check & Auto-Download
# ============================================================
def ensure_model(model_repo: str, local_dir: str = MODEL_DIR):
    from pathlib import Path
    import os, subprocess

    model_path = Path(local_dir)
    if any(model_path.glob("**/*.safetensors")):
        print(f"✅ Model already present in {local_dir}")
        return local_dir

    print(f"📦 Downloading {model_repo} from Hugging Face...")
    hf_token = os.getenv("HUGGINGFACE_TOKEN")
    if not hf_token:
        raise EnvironmentError("❌ Missing HUGGINGFACE_TOKEN in .env")

    subprocess.run([
        "huggingface-cli", "download", model_repo,
        "--local-dir", local_dir,
        "--token", hf_token
    ], check=True)

    print(f"✅ Model downloaded to {local_dir}")
    return local_dir



def launch_vllm_server(model_dir: str, port: int = 8000):
    import socket, subprocess, torch

    # Detect free GPU memory
    total_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    utilization = 0.7 if total_mem <= 12 else 0.9
    dtype = "float16"

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    if sock.connect_ex(("127.0.0.1", port)) == 0:
        print(f"✅ vLLM already running on port {port}")
        return
    sock.close()

    print(f"🚀 Launching vLLM server for {VLLM_MODEL} on {total_mem:.1f} GB GPU …")

    subprocess.Popen([
        "python", "-m", "vllm.entrypoints.openai.api_server",
        "--model", model_dir,
        "--served-model-name", VLLM_MODEL,
        "--port", str(port),
        "--max-model-len", "2048",
        "--gpu-memory-utilization", str(utilization),
        "--dtype", dtype,
        "--enforce-eager",
        "--chat-template", "{% for m in messages %}{{ m['role'] }}: {{ m['content'] }}\n{% endfor %}Assistant:",
        "--chat-template-content-format", "string"
    ])
    print(f"✅ vLLM server starting at http://localhost:{port}")




# ============================================================
# 🧩 Ensure Model + Launch vLLM
# ============================================================
MODEL_PATH = ensure_model(VLLM_MODEL, MODEL_DIR)
launch_vllm_server(MODEL_PATH, VLLM_PORT)

# ============================================================
# 🧩 Load Tabular & Vision Models
# ============================================================
tab_model = load_tabular_artifacts(Path("m2_outputs"))
vmodel, vtf, vlabels, vdevice = load_vision_model(Path("plantDoc-Output/efficientnet_b3"))

# ============================================================
# 🤖 Agents
# ============================================================
router_agent = AssistantAgent(name="RouterAgent", system_message="Route input to correct agent.")
vision_agent = AssistantAgent(name="VisionAgent", system_message="Analyze crop/leaf image for diseases.")
tabular_agent = AssistantAgent(name="TabularAgent", system_message="Analyze soil and environmental data.")
voice_agent = AssistantAgent(name="VoiceAgent", system_message="Convert voice queries using Whisper ASR.")
text_agent = AssistantAgent(name="TextAgent", system_message="Answer agriculture-related text queries.")
reasoning_agent = AssistantAgent(name="ReasoningAgent", system_message="Synthesize agent outputs into one summary.")
tts_agent = AssistantAgent(name="TTSAgent", system_message="Convert final result into audio using gTTS.")

group = GroupChat(
    agents=[router_agent, vision_agent, tabular_agent, voice_agent, text_agent, reasoning_agent, tts_agent],
    messages=[], max_round=5
)
manager = GroupChatManager(group)

# ============================================================
# ⚙️ Core Handlers
# ============================================================
def handle_vision(image_path):
    top = predict_image(vmodel, vtf, vlabels, Path(image_path), vdevice)
    return {
        "mode": "image",
        "prediction": top[0][0],
        "top": top,
        "message": f"Detected: {top[0][0]}. Other possibilities: {top[1][0]}, {top[2][0]}."
    }

def handle_tabular(inputs):
    crop, top = predict_tabular(tab_model, inputs)
    return {
        "mode": "tabular",
        "prediction": crop,
        "top": top,
        "message": f"Predicted crop: {crop} | NPK={inputs['Nitrogen']}/{inputs['Phosphorus']}/{inputs['Potassium']}, pH={inputs['pH_Value']}, Temp={inputs['Temperature']}°C."
    }

def handle_voice(audio_path):
    txt = whisper_transcribe(audio_path, WHISPER_SIZE)
    return {"mode": "voice", "message": f"Transcribed query: {txt}"}

def handle_text(query):
    system = build_system_prompt(DEFAULT_LANG, "helpful")
    txt = vllm_chat(
        VLLM_MODEL, VLLM_BASE_URL, None,
        system, query, timeout=LLM_TIMEOUT
    )
    return {"mode": "text", "message": txt}

def handle_reasoning(context):
    system = build_system_prompt(DEFAULT_LANG, "concise")
    prompt = f"{system}\nCombine these agent outputs into one multilingual recommendation:\n{json.dumps(context, indent=2)}"
    txt = vllm_chat(
        VLLM_MODEL, VLLM_BASE_URL, None,
        system, prompt, timeout=LLM_TIMEOUT
    )
    return txt

def handle_tts(text):
    return _tts_if_possible(text, DEFAULT_LANG)

# ============================================================
# 🧠 Coordinator
# ============================================================
def run_multimodal_session(tabular=None, image=None, audio=None, text=None):
    context = {}
    if image: context["VisionAgent"] = handle_vision(image)
    if audio: context["VoiceAgent"] = handle_voice(audio)
    if tabular: context["TabularAgent"] = handle_tabular(tabular)
    if text: context["TextAgent"] = handle_text(text)

    reasoning = handle_reasoning(context)
    tts = handle_tts(reasoning)

    return {
        "mode": ",".join(context.keys()) or "none",
        "prediction": ", ".join([context[k].get("prediction", "") for k in context if "prediction" in context[k]]),
        "top_table": {k: context[k].get("top", []) for k in context},
        "answer": reasoning,
        "tts_path": tts
    }

def route_controller(tabular_inputs, image_path, text_q, voice_path, cfg, lang_code):
    return run_multimodal_session(
        tabular=tabular_inputs, image=image_path, audio=voice_path, text=text_q
    )

# ============================================================
# 🚀 Run App
# ============================================================
if __name__ == "__main__":
    build_interface(
        Path("m2_outputs"),
        Path("plantDoc-Output/efficientnet_b3"),
        Path("images"),
        VLLM_MODEL,
        VLLM_BASE_URL,
        None,
        7863,
        WHISPER_SIZE,
        LLM_TIMEOUT,
        DEFAULT_LANG
    )
