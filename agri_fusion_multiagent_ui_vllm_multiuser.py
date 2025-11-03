#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AgriFusion Multi-Agentic Smart Farming Assistant (vLLM Edition – Multi-User)
=============================================================================
✅ Same UI layout as Phi-1.5 version
✅ Multi-user isolation (per-session state)
✅ Shared vLLM backend (TinyLlama or Phi)
✅ Auto GPU utilization tuning (FP16)
"""

import os, json, subprocess, socket, time, uuid
from pathlib import Path
from autogen import AssistantAgent, GroupChat, GroupChatManager
from dotenv import load_dotenv
from vllm_chat import vllm_chat
import gradio as gr
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
    import torch
    total_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    utilization = 0.7 if total_mem <= 12 else 0.9
    dtype = "float16"

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    if sock.connect_ex(("127.0.0.1", port)) == 0:
        print(f"✅ vLLM already running on port {port}")
        return
    sock.close()

    print(f"🚀 Launching vLLM server for {VLLM_MODEL} ({total_mem:.1f} GB GPU, util={utilization}) …")

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
# 🧩 Load Tabular & Vision Models (shared)
# ============================================================
tab_model = load_tabular_artifacts(Path("m2_outputs"))
vmodel, vtf, vlabels, vdevice = load_vision_model(Path("plantDoc-Output/efficientnet_b3"))

# ============================================================
# 🧠 Session Management
# ============================================================
def start_session():
    """Generate a unique session ID for each user."""
    sid = str(uuid.uuid4())[:8]
    os.makedirs("sessions", exist_ok=True)
    open(f"sessions/{sid}.json", "a").close()
    print(f"🆕 Started new session: {sid}")
    return sid

# ============================================================
# 🤖 Core Agent Handlers
# ============================================================
def handle_vision(image_path):
    top = predict_image(vmodel, vtf, vlabels, Path(image_path), vdevice)
    return {
        "mode": "image",
        "prediction": top[0][0],
        "top": top,
        "message": f"Detected: {top[0][0]} | Others: {top[1][0]}, {top[2][0]}"
    }

def handle_tabular(inputs):
    crop, top = predict_tabular(tab_model, inputs)
    return {
        "mode": "tabular",
        "prediction": crop,
        "top": top,
        "message": f"Predicted crop: {crop} | N={inputs['Nitrogen']} P={inputs['Phosphorus']} K={inputs['Potassium']} | pH={inputs['pH_Value']} | Temp={inputs['Temperature']}°C"
    }

def handle_voice(audio_path):
    txt = whisper_transcribe(audio_path, WHISPER_SIZE)
    return {"mode": "voice", "message": txt}

def handle_text(query):
    system = build_system_prompt(DEFAULT_LANG, "helpful")
    txt = vllm_chat(VLLM_MODEL, VLLM_BASE_URL, None, system, query, timeout=LLM_TIMEOUT)
    return {"mode": "text", "message": txt}

def handle_reasoning(context):
    system = build_system_prompt(DEFAULT_LANG, "concise")
    prompt = f"{system}\nCombine these agent outputs into one multilingual recommendation:\n{json.dumps(context, indent=2)}"
    return vllm_chat(VLLM_MODEL, VLLM_BASE_URL, None, system, prompt, timeout=LLM_TIMEOUT)

def handle_tts(text):
    return _tts_if_possible(text, DEFAULT_LANG)

# ============================================================
# 🧩 Coordinator (multi-session aware)
# ============================================================
def run_multimodal_session(session_id, tabular=None, image=None, audio=None, text=None):
    print(f"🧑‍🌾 [Session {session_id}] Processing query...")
    context = {}
    if image: context["VisionAgent"] = handle_vision(image)
    if audio: context["VoiceAgent"] = handle_voice(audio)
    if tabular: context["TabularAgent"] = handle_tabular(tabular)
    if text: context["TextAgent"] = handle_text(text)

    reasoning = handle_reasoning(context)
    tts = handle_tts(reasoning)
    print(f"✅ [Session {session_id}] Completed")

    with open(f"sessions/{session_id}.json", "a") as f:
        f.write(json.dumps({"context": context, "result": reasoning}, indent=2) + "\n")

    return {
        "mode": ",".join(context.keys()) or "none",
        "prediction": ", ".join([context[k].get("prediction", "") for k in context if "prediction" in context[k]]),
        "top_table": {k: context[k].get("top", []) for k in context},
        "answer": reasoning,
        "tts_path": tts
    }

def route_controller(tabular_inputs, image_path, text_q, voice_path, cfg, lang_code):
    session_id = start_session()  # assign unique session for each user
    print(f"🪪 Active session: {session_id}")
    return run_multimodal_session(
        session_id=session_id,
        tabular=tabular_inputs, image=image_path, audio=voice_path, text=text_q
    )

# ============================================================
# 🚀 Run App — with per-user isolation
# ============================================================
if __name__ == "__main__":
    app = build_interface(
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

    print("🌾 Launching AgriFusion Multi-User Assistant...")
    app.launch(server_name="0.0.0.0", server_port=7863, share=False)
