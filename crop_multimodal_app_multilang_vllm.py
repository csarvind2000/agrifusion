#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AgroSense Studio (vLLM) — Multilingual (EN/KN/TA/HI/ML/TE)
Tabular • Image • Text • Voice  |  Gradio 3.x-compatible UI
vLLM (OpenAI-compatible server) version — Ollama code stays untouched elsewhere.

Run vLLM server (example):
  python -m vllm.entrypoints.openai.api_server \
      --model meta-llama/Llama-3.1-8B-Instruct \
      --dtype auto --gpu-memory-utilization 0.90 --port 8000

Run app (example):
  python crop_multimodal_app_multilang_vllm.py \
      --tabular_model_dir m2_outputs \
      --vision_model_dir plantDoc-Output/efficientnet_b3 \
      --images_dir images \
      --vllm_model meta-llama/Llama-3.1-8B-Instruct \
      --vllm_base_url http://127.0.0.1:8000/v1 \
      --whisper_size base \
      --port 7863 \
      --default_lang hi
"""

import argparse, json, os, tempfile
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple

import numpy as np
import pandas as pd
import requests
import joblib
from PIL import Image
import torch
import torch.nn as nn
import gradio as gr

# ---------- timm for PlantDoc models ----------
import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform

# ---------- Optional Whisper + gTTS ----------
try:
    import whisper
    WHISPER_OK = True
except Exception:
    WHISPER_OK = False

try:
    from gtts import gTTS
    GTTS_OK = True
except Exception:
    GTTS_OK = False

# ======================= Language Support =======================
LANGS = [
    ("en", "English"),
    ("kn", "ಕನ್ನಡ (Kannada)"),
    ("ta", "தமிழ் (Tamil)"),
    ("hi", "हिन्दी (Hindi)"),
    ("ml", "മലയാളം (Malayalam)"),
    ("te", "తెలుగు (Telugu)"),
]

UI = {
    "en": {
        "title": "🌿 AgriFusion — Multimodal AI Smart Farming Assistant (vLLM)",
        "inputs": "Inputs",
        "soil": "Soil Inputs",
        "climate": "Climate Inputs",
        "presets": "Sample presets (click a row)",
        "image": "Image (Plant Image)",
        "upload_image": "Upload crop/leaf image",
        "ask": "Ask a Question (one-shot)",
        "text_ph": "e.g., Which pesticide should I spray & when for apple rust?",
        "sample_qs": "Sample questions",
        "voice_q": "Voice question (record/upload)",
        "keep_chat": "Keep chat when running new input",
        "run": "Run",
        "reset": "Reset All",
        "results": "Results",
        "mode": "Routed Mode",
        "pred": "Prediction / Detected Class",
        "probs": "Top Probabilities",
        "preview": "Preview",
        "answer": "Answer / Recommendations",
        "transcript": "Transcript (for voice)",
        "tts": "Voice output (TTS, if available)",
        "chat": "Chat",
        "chat_msg": "Message",
        "chat_send_text": "Send Text",
        "chat_send_voice": "Send Voice",
        "chat_clear": "Clear Chat",
        "chat_tts": "TTS reply (if available)",
        "lang": "Response Language",
    },
    "hi": {
        "title": "🌿 AgriFusion — बहुभाषी स्मार्ट खेती सहायक (vLLM)",
        "inputs": "इनपुट",
        "soil": "मृदा इनपुट",
        "climate": "जलवायु इनपुट",
        "presets": "नमूना प्रीसेट (किसी पंक्ति पर क्लिक करें)",
        "image": "चित्र (पौधा/पत्ती)",
        "upload_image": "फसल/पत्ती की छवि अपलोड करें",
        "ask": "सवाल पूछें (वन-शॉट)",
        "text_ph": "उदा., एप्पल रस्ट के लिए कौन-सी दवा और कब छिड़कें?",
        "sample_qs": "नमूना प्रश्न",
        "voice_q": "आवाज़ में पूछें (रिकॉर्ड/अपलोड)",
        "keep_chat": "नए इनपुट पर चैट बनाए रखें",
        "run": "चलाएं",
        "reset": "रीसेट",
        "results": "परिणाम",
        "mode": "चयनित मोड",
        "pred": "अनुमान / पहचाना गया वर्ग",
        "probs": "शीर्ष संभावनाएँ",
        "preview": "पूर्वावलोकन",
        "answer": "उत्तर / सिफारिशें",
        "transcript": "ट्रांसक्रिप्ट (आवाज)",
        "tts": "वॉइस आउटपुट (यदि उपलब्ध)",
        "chat": "चैट",
        "chat_msg": "संदेश",
        "chat_send_text": "टेक्स्ट भेजें",
        "chat_send_voice": "वॉइस भेजें",
        "chat_clear": "चैट साफ करें",
        "chat_tts": "टीटीएस उत्तर",
        "lang": "उत्तर की भाषा",
    },
    "ta": {
        "title": "🌿 AgriFusion — பல்மொழி ச்மார்ட் விவசாய உதவியாளர் (vLLM)",
        "inputs": "உள்ளீடுகள்",
        "soil": "மண் தகவல்",
        "climate": "காலநிலை தகவல்",
        "presets": "மாதிரி முன்னமைவுகள் (வரியை கிளிக் செய்யவும்)",
        "image": "படம் (தாவரம்/இலை)",
        "upload_image": "பயிர்/இலை படத்தை பதிவேற்றவும்",
        "ask": "கேள்வி கேளுங்கள் (ஒருமுறை)",
        "text_ph": "உதா., ஆப்பிள் ரஸ்டிற்கான மருந்தும் நேரமும்?",
        "sample_qs": "மாதிரி கேள்விகள்",
        "voice_q": "குரல் கேள்வி (பதிவு/பதிவேற்றம்)",
        "keep_chat": "புதிய உள்ளீட்டில் உரையாடலை வைத்திருங்கள்",
        "run": "இயக்கு",
        "reset": "மீட்டமை",
        "results": "முடிவுகள்",
        "mode": "தேர்ந்தெடுத்த முறை",
        "pred": "முன்கணிப்பு / கண்டறியப்பட்ட வகை",
        "probs": "சிறந்த சாத்தியங்கள்",
        "preview": "முன்நோக்கு",
        "answer": "பதில் / பரிந்துரைகள்",
        "transcript": "எழுத்தாக்கம் (குரல்)",
        "tts": "குரல் வெளியீடு (இருந்தால்)",
        "chat": "உரையாடல்",
        "chat_msg": "செய்தி",
        "chat_send_text": "உரை அனுப்பு",
        "chat_send_voice": "குரல் அனுப்பு",
        "chat_clear": "அழி",
        "chat_tts": "TTS பதில்",
        "lang": "பதில் மொழி",
    },
    "kn": {
        "title": "🌿 AgriFusion — ಬಹುಭಾಷಾ ಸ್ಮಾರ್ಟ್ ಕೃಷಿ ಸಹಾಯಕ (vLLM)",
        "inputs": "ಇನ್‌ಪುಟ್‌ಗಳು",
        "soil": "ಮಣ್ಣಿನ ಇನ್‌ಪುಟ್",
        "climate": "ಹವಾಮಾನ ಇನ್‌ಪುಟ್",
        "presets": "ಮಾದರಿ ಪ್ರಿಸೆಟ್‌ಗಳು (ಒಂದು ಸಾಲು ಕ್ಲಿಕ್ ಮಾಡಿ)",
        "image": "ಚಿತ್ರ (ಬೆಳೆ/ಎಲೆ)",
        "upload_image": "ಬೆಳೆ/ಎಲೆ ಚಿತ್ರದ ಅಪ್‌ಲೋಡ್",
        "ask": "ಒಂದು ಪ್ರಶ್ನೆ ಕೇಳಿ (ಒಮ್ಮೆ)",
        "text_ph": "ಉದಾ., ಆಪಲ್ ರಸ್ಟ್‌ಗೆ ಯಾವ ಔಷಧಿ ಮತ್ತು ಯಾವಾಗ?",
        "sample_qs": "ಮಾದರಿ ಪ್ರಶ್ನೆಗಳು",
        "voice_q": "ಧ್ವನಿ ಪ್ರಶ್ನೆ (ರೆಕಾರ್ಡ್/ಅಪ್‌ಲೋಡ್)",
        "keep_chat": "ಹೊಸ ಇನ್‌ಪುಟ್‌ನಲ್ಲಿ ಚಾಟ್ ಉಳಿಸು",
        "run": "ರನ್",
        "reset": "ರೀಸೆಟ್",
        "results": "ಫಲಿತಾಂಶಗಳು",
        "mode": "ಆಯ್ಕೆಯಾದ ಮೋಡ್",
        "pred": "ಅನುಮಾನ / ಪತ್ತೆಯಾದ ವರ್ಗ",
        "probs": "ಟಾಪ್ ಸಾಧ್ಯತೆಗಳು",
        "preview": "ಪ್ರೀವ್ಯೂ",
        "answer": "ಉತ್ತರ / ಶಿಫಾರಸುಗಳು",
        "transcript": "ಲಿಖಿತ ರೂಪ (ಧ್ವನಿ)",
        "tts": "ಧ್ವನಿ ಔಟ್‌ಪುಟ್ (ಇದ್ದರೆ)",
        "chat": "ಚಾಟ್",
        "chat_msg": "ಸಂದೇಶ",
        "chat_send_text": "ಟೆಕ್ಸ್ಟ್ ಕಳುಹಿಸು",
        "chat_send_voice": "ಧ್ವನಿ ಕಳುಹಿಸು",
        "chat_clear": "ಚಾಟ್ ಕ್ಲೀರ್",
        "chat_tts": "TTS ಉತ್ತರ",
        "lang": "ಉತ್ತರ ಭಾಷೆ",
    },
    "ml": {
        "title": "🌿 AgriFusion — ബഹുഭാഷാ സ്മാർട്ട് കൃഷി സഹായി (vLLM)",
        "inputs": "ഇൻപുട്ടുകൾ",
        "soil": "മണ്ണ് ഇൻപുട്ടുകൾ",
        "climate": "കാലാവസ്ഥ ഇൻപുട്ടുകൾ",
        "presets": "സാമ്പിൾ പ്രിസെറ്റുകൾ (ഒരു വരി ക്ലിക്കുചെയ്യുക)",
        "image": "ചിത്രം (സസ്യം/ഇല)",
        "upload_image": "വിള/ഇല ചിത്രം അപ്പ്‌ലോഡ് ചെയ്യുക",
        "ask": "ഒരു ചോദ്യം ചോദിക്കുക (ഒറ്റ തവണ)",
        "text_ph": "ഉദാ., ആപ്പിൾ റസ്റ്റ് – ഏത് മരുന്ന്, എപ്പോൾ?",
        "sample_qs": "സാമ്പിൾ ചോദ്യങ്ങൾ",
        "voice_q": "വോയിസ് ചോദ്യം (റെക്കോർഡ്/അപ്‌ലോഡ്)",
        "keep_chat": "പുതിയ ഇൻപുട്ടിൽ ചാറ്റ് നിലനിർത്തുക",
        "run": "റൺ",
        "reset": "റിസെറ്റ്",
        "results": "ഫലങ്ങൾ",
        "mode": "തിരഞ്ഞെടുത്ത മോഡ്",
        "pred": "ഭാവിഷ്യത്ത് / കണ്ടെത്തിയ വർഗ്ഗം",
        "probs": "മുൻനിര സാധ്യതകൾ",
        "preview": "പ്രിവ്യൂ",
        "answer": "ഉത്തരം / ശുപാർശകൾ",
        "transcript": "ട്രാൻസ്ക്രിപ്റ്റ് (വോയിസ്)",
        "tts": "വോയിസ് ഔട്ട്പുട്ട് (ലഭ്യമെങ്കിൽ)",
        "chat": "ചാറ്റ്",
        "chat_msg": "സന്ദേശം",
        "chat_send_text": "ടെക്സ്റ്റ് അയയ്ക്കുക",
        "chat_send_voice": "వോയిస్ അയയ്ക്കുക",
        "chat_clear": "ചാറ്റ് ക്ലിയർ",
        "chat_tts": "TTS മറുപടി",
        "lang": "മറുപടി ഭാഷ",
    },
    "te": {
        "title": "🌿 AgriFusion — బహుభాషా స్మార్ట్ వ్యవసాయ సహాయకుడు (vLLM)",
        "inputs": "ఇన్‌పుట్లు",
        "soil": "మట్టి ఇన్‌పుట్లు",
        "climate": "వాతావరణ ఇన్‌పుట్లు",
        "presets": "సాంపిల్ ప్రీసెట్లు (వరుసపై క్లిక్ చేయండి)",
        "image": "చిత్రం (పంట/ఆకు)",
        "upload_image": "పంట/ఆకు బొమ్మను అప్‌లోడ్ చేయండి",
        "ask": "ఒక ప్రశ్న అడగండి (వన్-షాట్)",
        "text_ph": "ఉదా., ఆపిల్ రస్ట్‌కు ఏ ఔషధం, ఎప్పుడు?",
        "sample_qs": "సాంపిల్ ప్రశ్నలు",
        "voice_q": "వాయిస్ ప్రశ్న (రికార్డ్/అప్‌లోడ్)",
        "keep_chat": "కొత్త ఇన్‌పుట్‌లో చాట్‌ను ఉంచండి",
        "run": "రన్",
        "reset": "రీసెట్",
        "results": "ఫలితాలు",
        "mode": "ఎంచుకున్న మోడ్",
        "pred": "అంచనా / గుర్తించిన తరగతి",
        "probs": "ఉన్నత సంభావ్యాలు",
        "preview": "ప్రివ్యూ",
        "answer": "సమాధానం / సిఫార్సులు",
        "transcript": "ట్రాన్స్క్రిప్ట్ (వాయిస్)",
        "tts": "వాయిస్ అవుట్‌పుట్ (ఉంటే)",
        "chat": "చాట్",
        "chat_msg": "సందేశం",
        "chat_send_text": "టెక్స్ట్ పంపండి",
        "chat_send_voice": "వాయిస్ పంపండి",
        "chat_clear": "చాట్ క్లియర్",
        "chat_tts": "TTS సమాధానం",
        "lang": "సమాధానం భాష",
    },
}

GTTS_LANG = {"en": "en", "kn": "kn", "ta": "ta", "hi": "hi", "ml": "ml", "te": "te"}

def ui_t(lang: str, key: str) -> str:
    if lang in UI and key in UI[lang]:
        return UI[lang][key]
    return UI["en"].get(key, key)

def build_system_prompt(lang_code: str, persona: str = "concise"):
    persona_line = {
        "concise": "You are a concise agronomy expert. Reply in the selected language with short, practical tips.",
        "helpful": "You are a helpful agronomy assistant. Reply in the selected language with short, clear points.",
    }[persona]

    lang_line = {
        "en": "Respond in English.",
        "kn": "Respond in Kannada. ಉತ್ತರವನ್ನು ಕನ್ನಡದಲ್ಲೇ ನೀಡಿ.",
        "ta": "Respond in Tamil. பதில் தமிழிலேயே கொடுக்கவும்.",
        "hi": "Respond in Hindi. उत्तर हिन्दी में ही दें.",
        "ml": "Respond in Malayalam. മറുപടി മലയാളത്തിലായിരിക്കണം.",
        "te": "Respond in Telugu. సమాధానం తెలుగులో ఇవ్వండి.",
    }.get(lang_code, "Respond in English.")

    style_line = (
        "Keep bullets tight (4–7 points). Avoid long paragraphs. "
        "Include measurements or schedules only briefly."
    )
    return f"{persona_line}\n{lang_line}\n{style_line}"

# ======================= Tabular models =======================
NUMERIC_FEATURES = ["Nitrogen","Phosphorus","Potassium","Temperature","Humidity","pH_Value","Rainfall"]

class MLPNet(nn.Module):
    def __init__(self, in_dim: int, n_classes: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 128), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(128, 64),     nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(64, n_classes)
        )
    def forward(self, x): return self.net(x)

class LSTMNet(nn.Module):
    def __init__(self, in_dim: int, n_classes: int, hid=64):
        super().__init__()
        self.embed = nn.Linear(1, 32)
        self.lstm  = nn.LSTM(input_size=32, hidden_size=hid, num_layers=1, batch_first=True)
        self.fc    = nn.Linear(hid, n_classes)
    def forward(self, x):
        b, f = x.shape
        x = x.view(b, f, 1)
        x = self.embed(x)
        _, (h, _) = self.lstm(x)
        return self.fc(h[-1])

class TransformerTab(nn.Module):
    def __init__(self, in_dim: int, n_classes: int, d_model=64, nhead=4, nlayers=2, dim_ff=128, dropout=0.1):
        super().__init__()
        self.value_proj = nn.Linear(1, d_model)
        self.feat_embed = nn.Parameter(torch.randn(in_dim, d_model))
        enc = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_ff,
                                         dropout=dropout, batch_first=True, activation="gelu")
        self.encoder = nn.TransformerEncoder(enc, num_layers=nlayers)
        self.norm = nn.LayerNorm(d_model)
        self.cls  = nn.Linear(d_model, n_classes)
    def forward(self, x):
        b, f = x.shape
        x = x.view(b, f, 1)
        v = self.value_proj(x)
        e = self.feat_embed.unsqueeze(0).expand(b, -1, -1)
        h = self.encoder(v + e)
        h = self.norm(h.mean(dim=1))
        return self.cls(h)

def apply_standardizer(df: pd.DataFrame, stats: Dict[str, Dict[str, float]]) -> np.ndarray:
    arr = []
    for c in NUMERIC_FEATURES:
        val = pd.to_numeric(df[c], errors="coerce").astype(float)
        m = stats[c]["mean"]; s = stats[c]["std"] if stats[c]["std"] != 0 else 1.0
        arr.append((val.fillna(m).values - m) / s)
    return np.vstack(arr).T

def load_tabular_artifacts(model_dir: Path):
    le = joblib.load(model_dir / "label_encoder.joblib")
    classes = json.load(open(model_dir / "classes.json"))
    model_type = open(model_dir / "model_type.txt").read().strip()
    scaler = json.load(open(model_dir / "scaler.json")) if (model_dir / "scaler.json").exists() else None

    if model_type == "sklearn":
        pipe = joblib.load(model_dir / "best_model.joblib")
        return {"type": "sklearn", "pipe": pipe, "le": le, "classes": classes, "scaler": scaler}
    else:
        meta = json.load(open(model_dir / "model_meta.json"))
        arch = meta["arch"]; in_dim = meta["in_dim"]; n_classes = meta["n_classes"]
        if arch == "MLP":
            model = MLPNet(in_dim, n_classes)
        elif arch == "LSTM":
            model = LSTMNet(in_dim, n_classes, hid=64)
        elif arch == "Transformer":
            model = TransformerTab(in_dim, n_classes, d_model=64, nhead=4, nlayers=2, dim_ff=128, dropout=0.1)
        else:
            raise RuntimeError(f"Unknown DL arch: {arch}")
        state = torch.load(model_dir / "best_model.pt", map_location="cpu")
        model.load_state_dict(state); model.eval()
        return {"type": "dl", "model": model, "le": le, "classes": classes, "scaler": scaler, "arch": arch}

def predict_tabular(loaded, row: Dict[str, float]):
    dfX = pd.DataFrame([row], columns=NUMERIC_FEATURES)
    if loaded["type"] == "sklearn":
        pipe = loaded["pipe"]
        try: proba = pipe.predict_proba(dfX)
        except Exception: proba = None
        pred_idx = pipe.predict(dfX)
    else:
        X_std = apply_standardizer(dfX, loaded["scaler"])
        xt = torch.tensor(X_std, dtype=torch.float32)
        with torch.no_grad():
            logits = loaded["model"](xt)
            proba_t = torch.softmax(logits, dim=1)
            proba = proba_t.numpy()
        pred_idx = np.argmax(proba, axis=1)
    pred_lbl = loaded["le"].inverse_transform(pred_idx)[0]
    classes = np.array(loaded["classes"])
    if proba is not None:
        order = np.argsort(-proba[0])[:2]
        top = [(classes[i], float(proba[0,i])) for i in order]
    else:
        top = [(pred_lbl, 1.0)]
    return pred_lbl, top

# ======================= Vision (timm) =======================
def _model_requires_img_size_arg(model_name: str) -> bool:
    name = model_name.lower()
    return any(k in name for k in ["vit_", "deit", "tnt_", "levit", "beit", "eva", "flexivit"])

def _build_timm_eval_transform(model, img_size: int):
    cfg = resolve_data_config({'img_size': img_size}, model=model)
    return create_transform(
        input_size=cfg['input_size'],
        interpolation=cfg['interpolation'],
        mean=cfg['mean'], std=cfg['std'],
        crop_pct=cfg.get('crop_pct', None),
        is_training=False,
    )

def load_vision_model(vision_dir: Path):
    meta = json.load(open(vision_dir/"vision_meta.json"))
    arch = meta["arch"]; img_size = int(meta["img_size"]); n = int(meta["num_classes"])
    label_map = json.load(open(vision_dir/"label_map.json"))
    inv_label = {int(k): v for k, v in label_map.items()}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if _model_requires_img_size_arg(arch):
        model = timm.create_model(arch, pretrained=False, num_classes=n, img_size=img_size)
    else:
        model = timm.create_model(arch, pretrained=False, num_classes=n)

    state = torch.load(vision_dir/"best_model.pt", map_location=device)
    model.load_state_dict(state, strict=True)
    model.eval().to(device)

    eval_tf = _build_timm_eval_transform(model, img_size)
    return model, eval_tf, inv_label, device

@torch.no_grad()
def predict_image(model, tf, inv_label: Dict[int,str], image_path: Path, device: torch.device):
    img = Image.open(image_path).convert("RGB")
    x = tf(img).unsqueeze(0).to(device)
    logits = model(x)
    prob = torch.softmax(logits, dim=1).detach().cpu().numpy()[0]
    top_idx = np.argsort(-prob)[:3]
    top = [(inv_label[int(i)], float(prob[i])) for i in top_idx]
    return top

# ======================= vLLM Chat (OpenAI-compatible) =======================
def vllm_chat(vllm_model: str, base_url: str, api_key: Optional[str], system: str, user: str, timeout=30) -> str:
    """
    Calls vLLM's OpenAI-compatible /chat/completions endpoint.
    base_url example: http://127.0.0.1:8000/v1
    """
    try:
        url = base_url.rstrip("/") + "/chat/completions"
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        payload = {
            "model": vllm_model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user",   "content": user},
            ],
            "temperature": 0.2,
            "top_p": 0.9,
            "stream": False,
        }
        r = requests.post(url, headers=headers, json=payload, timeout=timeout)
        if r.status_code == 200:
            data = r.json()
            msg = (data.get("choices",[{}])[0].get("message",{}) or {}).get("content","").strip()
            return msg or "(empty vLLM reply)"
        return f"(vLLM status {r.status_code}; {r.text[:200]})"
    except Exception as e:
        return f"(vLLM error: {e})"

def whisper_transcribe(audio_path: str, model_size="base"):
    if not WHISPER_OK:
        return "(Whisper not installed)"
    try:
        model = whisper.load_model(model_size)
        result = model.transcribe(audio_path)
        return result.get("text","").strip()
    except Exception as e:
        return f"(Whisper error: {e})"

# ======================= Routing helpers =======================
def _tabular_is_active(tab: Optional[Dict[str, float]]) -> bool:
    if not tab:
        return False
    if not all(k in tab and tab[k] is not None for k in NUMERIC_FEATURES):
        return False
    nonzero = any(abs(float(tab[k])) > 1e-9 for k in NUMERIC_FEATURES)
    return nonzero

def _tts_if_possible(text: str, lang_code: str) -> Optional[str]:
    if not GTTS_OK or not text or text.startswith("("):
        return None
    try:
        out = tempfile.mktemp(suffix=".mp3")
        gTTS(text, lang=GTTS_LANG.get(lang_code, "en")).save(out)
        return out
    except Exception:
        return None

def route_controller(tabular_inputs, image_path, text_q, voice_path, cfg, lang_code: str):
    """Priority: Image → Voice → Text → Tabular."""
    out = {
        "mode": "",
        "prediction": "",
        "top_table": pd.DataFrame(columns=["Rank", "Class", "Prob"]),
        "image": None,
        "answer": "",
        "transcript": "",
        "tts_path": None,
        "context": {},
    }

    def localize_prompt(prompt):
        return f"Please reply entirely in {dict(LANGS)[lang_code]}. {prompt}"

    # Image mode
    if image_path:
        out["mode"] = "image"
        top = predict_image(cfg["vision_model"], cfg["vision_tf"], cfg["vision_labels"], Path(image_path), cfg["vision_device"])
        rows = [{"Rank": f"Top-{i+1}", "Class": c, "Prob": round(p, 4)} for i, (c, p) in enumerate(top)]
        out["top_table"] = pd.DataFrame(rows)
        out["prediction"] = top[0][0]
        out["image"] = Image.open(image_path).convert("RGB")

        user = (
            f"Detected class: {top[0][0]}\nTop candidates: "
            f"{', '.join([f'{c}({p:.2f})' for c, p in top])}\n"
            "Give 4–6 agronomy pointers in bullet form."
        )
        system = build_system_prompt(lang_code, persona="concise")
        user = localize_prompt(user)
        out["answer"] = vllm_chat(cfg["vllm_model"], cfg["vllm_base_url"], cfg["vllm_api_key"], system, user, timeout=cfg["llm_timeout"])
        out["tts_path"] = _tts_if_possible(out["answer"], lang_code)
        out["context"] = {"mode": "image", "prediction": out["prediction"], "top": rows, "recommendations": out["answer"]}
        return out

    # Voice mode
    if voice_path:
        out["mode"] = "voice"
        transcript = whisper_transcribe(voice_path, cfg["whisper_size"])
        out["transcript"] = transcript
        if transcript.startswith("(") and "error" in transcript.lower():
            out["answer"] = transcript
            out["context"] = {"mode": "voice", "prediction": "", "top": [], "recommendations": out["answer"]}
            return out
        system = build_system_prompt(lang_code, persona="helpful")
        transcript = localize_prompt(transcript)
        out["answer"] = vllm_chat(cfg["vllm_model"], cfg["vllm_base_url"], cfg["vllm_api_key"], system, transcript, timeout=cfg["llm_timeout"])
        out["tts_path"] = _tts_if_possible(out["answer"], lang_code)
        out["context"] = {"mode": "voice", "prediction": "", "top": [], "recommendations": out["answer"]}
        return out

    # Text mode
    if text_q and text_q.strip():
        out["mode"] = "text"
        system = build_system_prompt(lang_code, persona="helpful")
        text_q = localize_prompt(text_q.strip())
        out["answer"] = vllm_chat(cfg["vllm_model"], cfg["vllm_base_url"], cfg["vllm_api_key"], system, text_q, timeout=cfg["llm_timeout"])
        out["tts_path"] = _tts_if_possible(out["answer"], lang_code)
        out["context"] = {"mode": "text", "prediction": "", "top": [], "recommendations": out["answer"]}
        return out

    # Tabular mode
    if _tabular_is_active(tabular_inputs):
        out["mode"] = "tabular"
        crop, top = predict_tabular(cfg["tab_model"], tabular_inputs)
        rows = [{"Rank": f"Top-{i+1}", "Class": c, "Prob": round(p, 4)} for i, (c, p) in enumerate(top)]
        out["top_table"] = pd.DataFrame(rows)
        out["prediction"] = crop
        user = (
            f"Predicted crop: {crop}\nProfile: N={tabular_inputs['Nitrogen']} P={tabular_inputs['Phosphorus']} "
            f"K={tabular_inputs['Potassium']}, pH={tabular_inputs['pH_Value']}, Temp={tabular_inputs['Temperature']}°C, "
            f"Humidity={tabular_inputs['Humidity']}%, Rainfall={tabular_inputs['Rainfall']} mm.\n"
            "Write 5–7 concise bullet recommendations about NPK, pH, irrigation, crop care, and tips."
        )
        system = build_system_prompt(lang_code, persona="concise")
        user = localize_prompt(user)
        out["answer"] = vllm_chat(cfg["vllm_model"], cfg["vllm_base_url"], cfg["vllm_api_key"], system, user, timeout=cfg["llm_timeout"])
        out["tts_path"] = _tts_if_possible(out["answer"], lang_code)
        out["context"] = {"mode": "tabular", "prediction": out["prediction"], "top": rows, "recommendations": out["answer"]}
        return out

    out["mode"] = "none"
    out["answer"] = "Please provide image OR soil/climate values OR a question (text or voice)."
    out["context"] = {"mode": "none", "prediction": "", "top": [], "recommendations": out["answer"]}
    return out

# ======================= UI =======================
CUSTOM_CSS = """
:root { --bg:#E8F5E9; --card:#ffffff; --accent:#1B5E20; }
:root {
  --bg: #D0E8D0;
  --card: #ffffff;
  --accent: #145A1F;
}
.gradio-container {background: var(--bg);}
h1.title { text-align:center; color:var(--accent); font-weight:800; margin-bottom:18px; }
.card { background:var(--card); padding:14px; border-radius:12px; box-shadow:0 2px 10px rgba(0,0,0,0.06); }
.compact .gr-form { gap: 8px !important; }
.df-small { max-height: 180px; overflow:auto; }
"""

def build_interface(tabular_model_dir: Path, vision_model_dir: Path, images_dir: Path,
                    vllm_model: str, vllm_base_url: str, vllm_api_key: Optional[str],
                    port: int, whisper_size: str, llm_timeout: int,
                    default_lang: str):

    # ---- Load models
    tab_model = load_tabular_artifacts(tabular_model_dir)
    vmodel, vtf, vlabels, vdevice = load_vision_model(vision_model_dir)

    cfg = {
        "tab_model": tab_model,
        "vision_model": vmodel,
        "vision_tf": vtf,
        "vision_labels": vlabels,
        "vision_device": vdevice,
        "vllm_model": vllm_model,
        "vllm_base_url": vllm_base_url.rstrip("/"),
        "vllm_api_key": vllm_api_key or os.getenv("VLLM_API_KEY") or None,
        "whisper_size": whisper_size,
        "llm_timeout": int(llm_timeout),
    }

    with gr.Blocks(css=CUSTOM_CSS, title="AgroSense Studio — AI Crop Assistant (vLLM)") as demo:
        lang_dd = gr.Dropdown(label=ui_t(default_lang, "lang"),
                              choices=[f"{code} — {name}" for code, name in LANGS],
                              value=f"{default_lang} — {dict(LANGS)[default_lang]}")

        title_html = gr.HTML(f"<h1 class='title'>{ui_t(default_lang, 'title')}</h1>")

        # Hidden states
        flags_json  = gr.Textbox(value=json.dumps(
            {"image_fresh": False, "voice_fresh": False, "text_fresh": False, "tabular_fresh": False}
        ), visible=False)
        ctx_json    = gr.Textbox(value=json.dumps({"mode":"none","prediction":"","top":[],"recommendations":""}), visible=False)

        def _lang_code_from_choice(choice: str) -> str:
            try:
                return choice.split(" — ")[0].strip()
            except Exception:
                return default_lang

        def _refresh_labels(choice: str):
            code = _lang_code_from_choice(choice)
            return (
                f"<h1 class='title'>{ui_t(code, 'title')}</h1>",
                gr.update(label=ui_t(code, "lang")),
                gr.update(value="### " + ui_t(code, "inputs")),
                gr.update(value="**" + ui_t(code, "soil") + "**"),
                gr.update(value="**" + ui_t(code, "climate") + "**"),
                gr.update(value="**" + ui_t(code, "presets") + "**"),
                gr.update(label=ui_t(code, "upload_image")),
                gr.update(value="**" + ui_t(code, "ask") + "**"),
                gr.update(placeholder=ui_t(code, "text_ph")),
                gr.update(value="**" + ui_t(code, "sample_qs") + "**"),
                gr.update(label=ui_t(code, "voice_q")),
                gr.update(label=ui_t(code, "keep_chat")),
                gr.update(value="### " + ui_t(code, "results")),
                gr.update(label=ui_t(code, "mode")),
                gr.update(label=ui_t(code, "pred")),
                gr.update(label=ui_t(code, "probs")),
                gr.update(label=ui_t(code, "preview")),
                gr.update(label=ui_t(code, "answer")),
                gr.update(label=ui_t(code, "transcript")),
                gr.update(label=ui_t(code, "tts")),
                gr.update(value="### " + ui_t(code, "chat")),
                gr.update(label=ui_t(code, "chat_msg")),
                gr.update(value=ui_t(code, "chat_send_text")),
                gr.update(value=ui_t(code, "chat_send_voice")),
                gr.update(value=ui_t(code, "chat_clear")),
                gr.update(value=ui_t(code, "run")),
                gr.update(value=ui_t(code, "reset")),
                gr.update(value="### " + ui_t(code, "image")),
                gr.update(value="### " + ui_t(code, "results")),
            )

        with gr.Row():
            # LEFT: Inputs
            with gr.Column(scale=1, elem_classes=["card","compact"]):
                inputs_md  = gr.Markdown("### " + ui_t(default_lang, "inputs"))
                soil_md    = gr.Markdown("**" + ui_t(default_lang, "soil") + "**")
                with gr.Row():
                    N   = gr.Number(label="Nitrogen (N)", value=None)
                    P   = gr.Number(label="Phosphorus (P)", value=None)
                    K   = gr.Number(label="Potassium (K)", value=None)
                    pHv = gr.Number(label="pH", value=None)
                climate_md = gr.Markdown("**" + ui_t(default_lang, "climate") + "**")
                with gr.Row():
                    T   = gr.Number(label="Temperature (°C)", value=None)
                    H   = gr.Number(label="Humidity (%)", value=None)
                    R   = gr.Number(label="Rainfall (mm)", value=None)

                presets_md = gr.Markdown("**" + ui_t(default_lang, "presets") + "**")
                presets = [
                    [90, 42, 43, 26, 85, 6.5, 220],
                    [80, 40, 40, 26, 75, 6.5, 120],
                    [50, 50, 50, 24, 60, 6.8, 80],
                    [20, 30, 20, 22, 55, 6.2, 90],
                    [30, 50,120, 28, 70, 6.0, 150],
                ]
                gr.Examples(examples=presets, inputs=[N,P,K,T,H,pHv,R])

                image_md = gr.Markdown("**" + ui_t(default_lang, "image") + "**")
                img_in = gr.Image(label=ui_t(default_lang, "upload_image"), type="filepath")

                ask_md = gr.Markdown("**" + ui_t(default_lang, "ask") + "**")
                txt_q = gr.Textbox(label="Text", placeholder=ui_t(default_lang, "text_ph"))

                samples_md = gr.Markdown("**" + ui_t(default_lang, "sample_qs") + "**")
                sample_qs = [
                    "Which pesticide should I spray & when for apple rust?",
                    "Give a weekly schedule to manage tomato late blight.",
                    "How can I raise soil pH from 5.5 to 6.5 safely?",
                    "What irrigation plan fits sandy soil for maize this summer?",
                ]
                gr.Examples(examples=[[q] for q in sample_qs], inputs=[txt_q])

                mic_q = gr.Audio(label=ui_t(default_lang, "voice_q"),
                                 sources=["microphone","upload"], type="filepath")

                keep_chat = gr.Checkbox(label=ui_t(default_lang, "keep_chat"), value=False)
                run_btn   = gr.Button(ui_t(default_lang, "run"), variant="primary")
                reset_btn = gr.Button(ui_t(default_lang, "reset"))

            # MIDDLE: Results
            with gr.Column(scale=1, elem_classes=["card"]):
                results_md = gr.Markdown("### " + ui_t(default_lang, "results"))
                mode_out = gr.Textbox(label=ui_t(default_lang, "mode"), interactive=False)
                pred_out = gr.Textbox(label=ui_t(default_lang, "pred"), interactive=False)

                top_out  = gr.Dataframe(
                    value=pd.DataFrame(columns=["Rank","Class","Prob"]),
                    label=ui_t(default_lang, "probs"),
                    interactive=False,
                    elem_classes=["df-small"]
                )

                img_out  = gr.Image(label=ui_t(default_lang, "preview"), type="pil")
                reco_out = gr.Markdown(label=ui_t(default_lang, "answer"))
                tr_out   = gr.Textbox(label=ui_t(default_lang, "transcript"), interactive=False)
                tts_out  = gr.Audio(label=ui_t(default_lang, "tts"), interactive=False)

            # RIGHT: Chat
            with gr.Column(scale=1, elem_classes=["card"]):
                chat_md     = gr.Markdown("### " + ui_t(default_lang, "chat"))
                chat_box    = gr.Chatbot(label="Agronomy Assistant", height=450)
                chat_input  = gr.Textbox(label=ui_t(default_lang, "chat_msg"),
                                         placeholder=ui_t(default_lang, "text_ph"))
                chat_voice  = gr.Audio(label=ui_t(default_lang, "voice_q"),
                                       sources=["microphone","upload"], type="filepath")
                chat_send_text  = gr.Button(ui_t(default_lang, "chat_send_text"), variant="primary")
                chat_send_voice = gr.Button(ui_t(default_lang, "chat_send_voice"))
                chat_clear      = gr.Button(ui_t(default_lang, "chat_clear"))
                chat_tts        = gr.Audio(label=ui_t(default_lang, "chat_tts"), interactive=False)

        # ---------- React to language changes ----------
        lang_dd.change(
            _refresh_labels,
            inputs=[lang_dd],
            outputs=[
                title_html, lang_dd, inputs_md, soil_md, climate_md, presets_md,
                img_in, ask_md, txt_q, samples_md, mic_q, keep_chat, results_md,
                mode_out, pred_out, top_out, img_out, reco_out, tr_out, tts_out,
                chat_md, chat_input, chat_send_text, chat_send_voice, chat_clear,
                run_btn, reset_btn
            ]
        )

        # ---------- Freshness flags ----------
        def _mark_flag(flags_str: str, key: str) -> str:
            flags = json.loads(flags_str or "{}")
            flags[key] = True
            return json.dumps(flags)

        img_in.change(lambda s: _mark_flag(s,"image_fresh"), inputs=[flags_json], outputs=[flags_json])
        mic_q.change(lambda s: _mark_flag(s,"voice_fresh"), inputs=[flags_json], outputs=[flags_json])
        txt_q.change(lambda s: _mark_flag(s,"text_fresh"),  inputs=[flags_json], outputs=[flags_json])
        for comp in (N,P,K,T,H,pHv,R):
            comp.change(lambda s: _mark_flag(s,"tabular_fresh"), inputs=[flags_json], outputs=[flags_json])

        # ---------- RUN ----------
        def on_run(n,p,k,t,h,ph,r,image_path,text_q,voice_path,flags_str,ctx_str,keep_chat_flag,lang_choice):
            try: flags = json.loads(flags_str or "{}")
            except Exception: flags = {"image_fresh":False,"voice_fresh":False,"text_fresh":False,"tabular_fresh":False}

            lang_code = (lang_choice.split(" — ")[0].strip()
                         if isinstance(lang_choice, str) else default_lang)

            if not flags.get("image_fresh"):  image_path = None
            if not flags.get("voice_fresh"):  voice_path = None
            if not flags.get("text_fresh"):   text_q = None
            if not flags.get("tabular_fresh"):
                n=p=k=t=h=ph=r=None

            tab = None
            if any(v is not None for v in [n,p,k,t,h,ph,r]):
                tab = {"Nitrogen":n, "Phosphorus":p, "Potassium":k,
                       "Temperature":t, "Humidity":h, "pH_Value":ph, "Rainfall":r}

            res = route_controller(tab, image_path, text_q, voice_path, cfg, lang_code)

            fresh_reset = json.dumps({"image_fresh":False,"voice_fresh":False,"text_fresh":False,"tabular_fresh":False})
            ctx_out = json.dumps(res["context"])
            chat_reset = [] if not keep_chat_flag else gr.update()

            return (res["mode"], res["prediction"], res["top_table"], res["image"],
                    res["answer"], res["transcript"], res["tts_path"],
                    fresh_reset, ctx_out, chat_reset)

        run_btn.click(
            on_run,
            inputs=[N,P,K,T,H,pHv,R,img_in,txt_q,mic_q,flags_json,ctx_json,keep_chat,lang_dd],
            outputs=[mode_out,pred_out,top_out,img_out,reco_out,tr_out,tts_out,flags_json,ctx_json,chat_box]
        )

        # ---------- Reset ----------
        def do_reset():
            return (None,None,None,None,None,None,None,
                    json.dumps({"image_fresh":False,"voice_fresh":False,"text_fresh":False,"tabular_fresh":False}),
                    json.dumps({"mode":"none","prediction":"","top":[],"recommendations":""}),
                    [], "", None)

        reset_btn.click(
            do_reset,
            inputs=[],
            outputs=[N,P,K,T,H,pHv,R,flags_json,ctx_json,chat_box,txt_q,chat_tts]
        )

        # ---------- Chat ----------
        def _chat_core(history: List[Tuple[str,str]], user_msg: str, ctx_str: str, lang_choice: str) -> Tuple[List[Tuple[str,str]], str, Optional[str]]:
            history = history or []
            if not user_msg or not user_msg.strip():
                return history, "", None
            try:
                context = json.loads(ctx_str or "{}")
            except Exception:
                context = {"mode":"none","prediction":"","top":[],"recommendations":""}

            lang_code = (lang_choice.split(" — ")[0].strip()
                         if isinstance(lang_choice, str) else default_lang)
            ctx_mode = context.get("mode","none")
            ctx_pred = context.get("prediction","")
            top_rows = context.get("top", [])
            top_str  = ", ".join([f"{r.get('Class','?')}({r.get('Prob','?')})" for r in top_rows[:3]]) if isinstance(top_rows,list) else "n/a"

            system = build_system_prompt(lang_code, persona="helpful") + \
                     f"\nLatest context → mode: {ctx_mode}; prediction: {ctx_pred}; top: {top_str}."
            try:
                reply = vllm_chat(cfg["vllm_model"], cfg["vllm_base_url"], cfg["vllm_api_key"], system, user_msg, timeout=cfg["llm_timeout"])
            except Exception as e:
                reply = f"(vLLM chat error: {e})"

            history = history + [(user_msg, reply)]
            tts_path = _tts_if_possible(reply, lang_code)
            return history, "", tts_path

        chat_send_text.click(
            lambda hist, text, ctx, langc: _chat_core(hist, text, ctx, langc),
            inputs=[chat_box, chat_input, ctx_json, lang_dd],
            outputs=[chat_box, chat_input, chat_tts]
        )

        def chat_voice_fn(hist, voice_path, ctx, langc):
            if not voice_path:
                return hist, "", None
            transcript = whisper_transcribe(voice_path, cfg["whisper_size"])
            if transcript.startswith("(") and "error" in transcript.lower():
                return (hist or []) + [("","(voice error)")], "", None
            return _chat_core(hist, transcript, ctx, langc)

        chat_send_voice.click(
            chat_voice_fn,
            inputs=[chat_box, chat_voice, ctx_json, lang_dd],
            outputs=[chat_box, chat_input, chat_tts]
        )

        chat_clear.click(lambda: ([], "", None), inputs=[], outputs=[chat_box, chat_input, chat_tts])

        demo.launch(server_name="0.0.0.0", server_port=port, share=False)

# ======================= CLI =======================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tabular_model_dir", required=True)
    ap.add_argument("--vision_model_dir", required=True)
    ap.add_argument("--images_dir", required=True)
    ap.add_argument("--vllm_model", required=True, help="HF model id served by vLLM (e.g., meta-llama/Llama-3.1-8B-Instruct)")
    ap.add_argument("--vllm_base_url", default="http://127.0.0.1:8000/v1", help="Base URL for vLLM OpenAI-compatible server")
    ap.add_argument("--vllm_api_key", default="", help="Optional API key if your vLLM server enforces auth")
    ap.add_argument("--whisper_size", default="base", help="tiny|base|small|medium|large")
    ap.add_argument("--port", type=int, default=7861)
    ap.add_argument("--llm_timeout", type=int, default=10, help="seconds to wait for vLLM before returning UI")
    ap.add_argument("--default_lang", default="en", choices=[c for c,_ in LANGS],
                    help="Default response language for UI + TTS")
    args = ap.parse_args()

    build_interface(
        Path(args.tabular_model_dir), Path(args.vision_model_dir), Path(args.images_dir),
        args.vllm_model, args.vllm_base_url, (args.vllm_api_key or None),
        args.port, args.whisper_size, args.llm_timeout, args.default_lang
    )

if __name__ == "__main__":
    main()
