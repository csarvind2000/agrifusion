#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AgriFusion Multi-Agentic Multilingual Smart Farming Assistant 🌾
================================================================
LLM: Ollama (llama3 or compatible)
Frontend: Gradio
Agents: Router, Vision, Voice, Tabular, Text, Reasoning, TTS
"""

import json, tempfile, requests
from pathlib import Path
import torch
from autogen import AssistantAgent, GroupChat, GroupChatManager
from crop_multimodal_app_multilang import (
    load_tabular_artifacts, predict_tabular,
    load_vision_model, predict_image,
    whisper_transcribe, build_system_prompt, _tts_if_possible,
    build_interface
)

# ============================================================
# Load model assets and configs
# ============================================================
OLLAMA_MODEL = "llama3:latest"
WHISPER_SIZE = "base"
DEFAULT_LANG = "en"
LLM_TIMEOUT = 20

tab_model = load_tabular_artifacts(Path("m2_outputs"))
vmodel, vtf, vlabels, vdevice = load_vision_model(Path("plantDoc-Output/efficientnet_b3"))

# ============================================================
# Define agents (Autogen framework)
# ============================================================
router_agent = AssistantAgent(
    name="RouterAgent",
    system_message="You are the Router Agent. Determine which input types are provided — image, voice, tabular, or text — and decide which specialist agents to activate."
)

vision_agent = AssistantAgent(
    name="VisionAgent",
    system_message="You are a plant pathology vision expert. Detect crop disease or plant condition from the uploaded image."
)

tabular_agent = AssistantAgent(
    name="TabularAgent",
    system_message="You are an agronomy expert analyzing soil, NPK, pH, humidity, and rainfall data to predict crops and suggest care recommendations."
)

voice_agent = AssistantAgent(
    name="VoiceAgent",
    system_message="You are a speech understanding assistant that converts farmer audio queries into text using Whisper ASR."
)

text_agent = AssistantAgent(
    name="TextAgent",
    system_message="You are a multilingual agronomy assistant answering general agriculture-related questions in simple farmer-friendly terms."
)

reasoning_agent = AssistantAgent(
    name="ReasoningAgent",
    system_message="You are the final synthesis agent. Combine outputs from all other agents into one short, multilingual, human-readable summary with actionable advice."
)

tts_agent = AssistantAgent(
    name="TTSAgent",
    system_message="You are a voice synthesis assistant converting final answers into speech using gTTS."
)

group = GroupChat(
    agents=[router_agent, vision_agent, tabular_agent, voice_agent, text_agent, reasoning_agent, tts_agent],
    messages=[],
    max_round=5
)
manager = GroupChatManager(group)

# ============================================================
# Core Agent Functions
# ============================================================

def handle_vision(image_path):
    top = predict_image(vmodel, vtf, vlabels, Path(image_path), vdevice)
    return {
        "mode": "image",
        "prediction": top[0][0],
        "top": top,
        "message": f"Detected issue: {top[0][0]}. Other possibilities: {top[1][0]}, {top[2][0]}."
    }

def handle_tabular(inputs):
    crop, top = predict_tabular(tab_model, inputs)
    return {
        "mode": "tabular",
        "prediction": crop,
        "top": top,
        "message": f"Predicted crop: {crop}. Based on NPK={inputs['Nitrogen']}/{inputs['Phosphorus']}/{inputs['Potassium']}, pH={inputs['pH_Value']}, T={inputs['Temperature']}°C."
    }

def handle_voice(audio_path):
    txt = whisper_transcribe(audio_path, WHISPER_SIZE)
    return {"mode": "voice", "message": f"Transcribed farmer query: {txt}"}

def handle_text(query):
    system = build_system_prompt(DEFAULT_LANG, "helpful")
    payload = {
        "model": OLLAMA_MODEL,
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": query}],
    }
    r = requests.post("http://127.0.0.1:11434/api/chat", json=payload, timeout=LLM_TIMEOUT)
    txt = (r.json().get("message", {}) or {}).get("content", "")
    return {"mode": "text", "message": txt}

def handle_reasoning(context):
    """Fuse all responses into one coherent multilingual recommendation"""
    system = build_system_prompt(DEFAULT_LANG, "concise")
    prompt = f"Combine the following multi-agent outputs into a single practical multilingual report for farmers:\n\n{json.dumps(context, indent=2)}"
    payload = {
        "model": OLLAMA_MODEL,
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": prompt}],
    }
    r = requests.post("http://127.0.0.1:11434/api/chat", json=payload, timeout=LLM_TIMEOUT)
    return (r.json().get("message", {}) or {}).get("content", "")

def handle_tts(text):
    path = _tts_if_possible(text, DEFAULT_LANG)
    return path

# ============================================================
# Multi-Agent Coordinator
# ============================================================

def run_multimodal_session(tabular=None, image=None, audio=None, text=None):
    """Autogen multi-agent orchestration"""
    context = {}
    if image:
        context["VisionAgent"] = handle_vision(image)
    if audio:
        context["VoiceAgent"] = handle_voice(audio)
    if tabular:
        context["TabularAgent"] = handle_tabular(tabular)
    if text:
        context["TextAgent"] = handle_text(text)

    reasoning = handle_reasoning(context)
    tts = handle_tts(reasoning)

    return {
        "mode": ",".join(context.keys()) if context else "none",
        "prediction": ", ".join([context[k].get("prediction", "") for k in context if "prediction" in context[k]]),
        "top_table": {k: context[k].get("top", []) for k in context},
        "answer": reasoning,
        "tts_path": tts
    }

# ============================================================
# Integrate into your existing Gradio UI
# ============================================================

def route_controller(tabular_inputs, image_path, text_q, voice_path, cfg, lang_code: str):
    """
    Replaces original single-agent route_controller with multi-agent orchestration
    """
    return run_multimodal_session(
        tabular=tabular_inputs,
        image=image_path,
        audio=voice_path,
        text=text_q
    )

# ============================================================
# Run the Gradio frontend
# ============================================================
if __name__ == "__main__":
    build_interface(
        Path("m2_outputs"),
        Path("plantDoc-Output/efficientnet_b3"),
        Path("images"),
        OLLAMA_MODEL,
        7863,
        WHISPER_SIZE,
        LLM_TIMEOUT,
        DEFAULT_LANG
    )
