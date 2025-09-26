#!/usr/bin/env python3
"""
AgroSense Studio — AI Crop Assistant (Tabular • Image • Text • Voice)
Compatibility-safe Gradio app (works on older Gradio 3.x)

Run:
  python crop_multimodal_app.py \
      --tabular_model_dir m2_outputs \
      --vision_model_dir plantDoc-Output/efficientnet_b3 \
      --images_dir images \
      --ollama_model llama3 \
      --whisper_size base \
      --port 7863
"""

import argparse, json, tempfile
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

# ======================= LLM / Whisper =======================
def ollama_chat(ollama_model: str, system: str, user: str, timeout=10) -> str:
    try:
        payload = {"model": ollama_model, "messages": [
            {"role":"system","content":system},
            {"role":"user","content":user},
        ], "stream": False}
        r = requests.post("http://localhost:11434/api/chat", json=payload, timeout=timeout)
        if r.status_code == 200:
            return (r.json().get("message",{}) or {}).get("content","").strip()
        return f"(LLM status {r.status_code}; showing prediction without tips.)"
    except Exception:
        return "(LLM unavailable; showing prediction without tips.)"

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

def route_controller(tabular_inputs, image_path, text_q, voice_path, cfg):
    """Priority: Image → Voice → Text → Tabular."""
    out = {
        "mode": "",
        "prediction": "",
        "top_table": pd.DataFrame(columns=["Rank","Class","Prob"]),
        "image": None,
        "answer": "",
        "transcript": "",
        "tts_path": None,
        "context": {},
    }

    # 1) Image
    if image_path:
        out["mode"] = "image"
        top = predict_image(cfg["vision_model"], cfg["vision_tf"], cfg["vision_labels"], Path(image_path), cfg["vision_device"])
        rows = [{"Rank": f"Top-{i+1}", "Class": c, "Prob": round(p,4)} for i,(c,p) in enumerate(top)]
        out["top_table"] = pd.DataFrame(rows)
        out["prediction"] = top[0][0]
        out["image"] = Image.open(image_path).convert("RGB")
        user = (f"Detected class: {top[0][0]}\n"
                f"Top candidates: {', '.join([f'{c}({p:.2f})' for c,p in top])}\n"
                "Give 4–6 very short agronomy pointers (Markdown bullets).")
        out["answer"] = ollama_chat(cfg["ollama_model"], "You are a concise agronomy expert.", user, timeout=cfg["llm_timeout"])
        if GTTS_OK and out["answer"]:
            f = tempfile.mktemp(suffix=".mp3"); gTTS(out["answer"]).save(f); out["tts_path"] = f
        out["context"] = {"mode":"image","prediction":out["prediction"],"top":rows,"recommendations":out["answer"]}
        return out

    # 2) Voice
    if voice_path:
        out["mode"] = "voice"
        transcript = whisper_transcribe(voice_path, cfg["whisper_size"])
        out["transcript"] = transcript
        if transcript.startswith("(") and "error" in transcript.lower():
            out["answer"] = transcript
            out["context"] = {"mode":"voice","prediction":"","top":[],"recommendations":out["answer"]}
            return out
        out["answer"] = ollama_chat(cfg["ollama_model"], "You are a helpful agronomy assistant.", transcript, timeout=cfg["llm_timeout"])
        if GTTS_OK and out["answer"]:
            f = tempfile.mktemp(suffix=".mp3"); gTTS(out["answer"]).save(f); out["tts_path"] = f
        out["context"] = {"mode":"voice","prediction":"","top":[],"recommendations":out["answer"]}
        return out

    # 3) Text
    if text_q and text_q.strip():
        out["mode"] = "text"
        out["answer"] = ollama_chat(cfg["ollama_model"], "You are a helpful agronomy assistant.", text_q.strip(), timeout=cfg["llm_timeout"])
        if GTTS_OK and out["answer"]:
            f = tempfile.mktemp(suffix=".mp3"); gTTS(out["answer"]).save(f); out["tts_path"] = f
        out["context"] = {"mode":"text","prediction":"","top":[],"recommendations":out["answer"]}
        return out

    # 4) Tabular
    if _tabular_is_active(tabular_inputs):
        out["mode"] = "tabular"
        crop, top = predict_tabular(cfg["tab_model"], tabular_inputs)
        rows = [{"Rank": f"Top-{i+1}", "Class": c, "Prob": round(p,4)} for i,(c,p) in enumerate(top)]
        out["top_table"] = pd.DataFrame(rows)
        out["prediction"] = crop
        user = (f"Predicted crop: {crop}\n"
                f"Profile: N={tabular_inputs['Nitrogen']} P={tabular_inputs['Phosphorus']} K={tabular_inputs['Potassium']}, "
                f"pH={tabular_inputs['pH_Value']}, Temp={tabular_inputs['Temperature']}°C, "
                f"Humidity={tabular_inputs['Humidity']}%, Rainfall={tabular_inputs['Rainfall']} mm\n\n"
                "Write 5–7 concise recommendations (Markdown): NPK & Timing; pH Management; Irrigation; Crop-Specific; Extra Tips.")
        out["answer"] = ollama_chat(cfg["ollama_model"], "You are an agronomy assistant.", user, timeout=cfg["llm_timeout"])
        if GTTS_OK and out["answer"]:
            f = tempfile.mktemp(suffix=".mp3"); gTTS(out["answer"]).save(f); out["tts_path"] = f
        out["context"] = {"mode":"tabular","prediction":out["prediction"],"top":rows,"recommendations":out["answer"]}
        return out

    out["mode"] = "none"
    out["answer"] = "Please provide image OR soil/climate values OR a question (text or voice)."
    out["context"] = {"mode":"none","prediction":"","top":[],"recommendations":out["answer"]}
    return out

# ======================= UI =======================
CUSTOM_CSS = """
:root { --bg:#E8F5E9; --card:#ffffff; --accent:#1B5E20; }
.gradio-container {background: var(--bg);}
h1.title { text-align:center; color:var(--accent); font-weight:800; margin-bottom:18px; }
.card { background:var(--card); padding:14px; border-radius:12px; box-shadow:0 2px 10px rgba(0,0,0,0.06); }
.compact .gr-form { gap: 8px !important; }
.df-small { max-height: 180px; overflow:auto; }
"""

def build_interface(tabular_model_dir: Path, vision_model_dir: Path, images_dir: Path,
                    ollama_model: str, port: int, whisper_size: str, llm_timeout: int):

    tab_model = load_tabular_artifacts(tabular_model_dir)
    vmodel, vtf, vlabels, vdevice = load_vision_model(vision_model_dir)

    cfg = {
        "tab_model": tab_model,
        "vision_model": vmodel,
        "vision_tf": vtf,
        "vision_labels": vlabels,
        "vision_device": vdevice,
        "ollama_model": ollama_model,
        "whisper_size": whisper_size,
        "llm_timeout": int(llm_timeout),
    }

    with gr.Blocks(css=CUSTOM_CSS, title="AgroSense Studio — AI Crop Assistant") as demo:
        gr.HTML("<h1 class='title'>🌿 AgriFusion — Multimodal AI Smart Farming Assistant</h1>")

        # Hidden JSON "states" for compatibility (avoid gr.State I/O issues on 3.x)
        flags_json  = gr.Textbox(value=json.dumps(
            {"image_fresh": False, "voice_fresh": False, "text_fresh": False, "tabular_fresh": False}
        ), visible=False)
        ctx_json    = gr.Textbox(value=json.dumps({"mode":"none","prediction":"","top":[],"recommendations":""}), visible=False)

        with gr.Row():
            # LEFT: Inputs
            with gr.Column(scale=1, elem_classes=["card","compact"]):
                gr.Markdown("### Inputs")
                gr.Markdown("**Soil & Climate (Tabular)**")

                gr.Markdown("**Soil Inputs**")
                with gr.Row():
                    N   = gr.Number(label="Nitrogen (N)", value=None)
                    P   = gr.Number(label="Phosphorus (P)", value=None)
                    K   = gr.Number(label="Potassium (K)", value=None)
                    pHv = gr.Number(label="pH", value=None)
                gr.Markdown("**Climate Inputs**")
                with gr.Row():
                    T   = gr.Number(label="Temperature (°C)", value=None)
                    H   = gr.Number(label="Humidity (%)", value=None)
                    R   = gr.Number(label="Rainfall (mm)", value=None)

                gr.Markdown("**Sample presets (click a row)**")
                presets = [
                    [90, 42, 43, 26, 85, 6.5, 220],
                    [80, 40, 40, 26, 75, 6.5, 120],
                    [50, 50, 50, 24, 60, 6.8, 80],
                    [20, 30, 20, 22, 55, 6.2, 90],
                    [30, 50,120, 28, 70, 6.0, 150],
                ]
                gr.Examples(examples=presets, inputs=[N,P,K,T,H,pHv,R])

                gr.Markdown("**Image (PlantImage)**")
                img_in = gr.Image(label="Upload crop/leaf image", type="filepath")

                gr.Markdown("**Ask a Question (one-shot)**")
                txt_q = gr.Textbox(label="Text question",
                                   placeholder="e.g., Which pesticide should I spray & when for apple rust?")
                gr.Markdown("**Sample questions**")
                sample_qs = [
                    "Which pesticide should I spray & when for apple rust?",
                    "Give a weekly schedule to manage tomato late blight.",
                    "How can I raise soil pH from 5.5 to 6.5 safely?",
                    "What irrigation plan fits sandy soil for maize this summer?",
                ]
                gr.Examples(examples=[[q] for q in sample_qs], inputs=[txt_q])

                mic_q = gr.Audio(label="Voice question (record/upload)",
                                 sources=["microphone","upload"], type="filepath")

                keep_chat = gr.Checkbox(label="Keep chat when running new input", value=False)
                run_btn   = gr.Button("Run", variant="primary")
                reset_btn = gr.Button("Reset All")

            # MIDDLE: Results
            with gr.Column(scale=1, elem_classes=["card"]):
                gr.Markdown("### Results")
                mode_out = gr.Textbox(label="Routed Mode", interactive=False)
                pred_out = gr.Textbox(label="Prediction / Detected Class", interactive=False)

                top_out  = gr.Dataframe(
                    value=pd.DataFrame(columns=["Rank","Class","Prob"]),
                    label="Top Probabilities",
                    interactive=False,
                    elem_classes=["df-small"]
                )

                img_out  = gr.Image(label="Preview", type="pil")
                reco_out = gr.Markdown(label="Answer / Recommendations")
                tr_out   = gr.Textbox(label="Transcript (for voice)", interactive=False)
                tts_out  = gr.Audio(label="Voice output (TTS, if available)", interactive=False)

            # RIGHT: Chat
            with gr.Column(scale=1, elem_classes=["card"]):
                gr.Markdown("### Chat")
                chat_box   = gr.Chatbot(label="Agronomy Assistant", height=450)
                chat_input = gr.Textbox(label="Message", placeholder="Type your question…")
                chat_voice = gr.Audio(label="Or speak (record/upload)", sources=["microphone","upload"], type="filepath")
                chat_send_text  = gr.Button("Send Text", variant="primary")
                chat_send_voice = gr.Button("Send Voice")
                chat_clear      = gr.Button("Clear Chat")
                chat_tts        = gr.Audio(label="TTS reply (if available)", interactive=False)

        # ---------- Freshness flag helpers (via hidden JSON textbox) ----------
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
        def on_run(n,p,k,t,h,ph,r,image_path,text_q,voice_path,flags_str,ctx_str,keep_chat_flag):
            # parse flags/context
            try: flags = json.loads(flags_str or "{}")
            except Exception: flags = {"image_fresh":False,"voice_fresh":False,"text_fresh":False,"tabular_fresh":False}

            # use only fresh inputs
            if not flags.get("image_fresh"):  image_path = None
            if not flags.get("voice_fresh"):  voice_path = None
            if not flags.get("text_fresh"):   text_q = None
            if not flags.get("tabular_fresh"):
                n=p=k=t=h=ph=r=None

            tab = None
            if any(v is not None for v in [n,p,k,t,h,ph,r]):
                tab = {"Nitrogen":n, "Phosphorus":p, "Potassium":k,
                       "Temperature":t, "Humidity":h, "pH_Value":ph, "Rainfall":r}

            res = route_controller(tab, image_path, text_q, voice_path, cfg)

            # reset flags each run
            fresh_reset = json.dumps({"image_fresh":False,"voice_fresh":False,"text_fresh":False,"tabular_fresh":False})
            ctx_out = json.dumps(res["context"])

            # If not keeping chat, clear it here by returning []
            chat_reset = [] if not keep_chat_flag else gr.update()

            return (res["mode"], res["prediction"], res["top_table"], res["image"],
                    res["answer"], res["transcript"], res["tts_path"],
                    fresh_reset, ctx_out, chat_reset)

        run_btn.click(
            on_run,
            inputs=[N,P,K,T,H,pHv,R,img_in,txt_q,mic_q,flags_json,ctx_json,keep_chat],
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

        # ---------- Chat (classic tuple history) ----------
        def _chat_core(history: List[Tuple[str,str]], user_msg: str, ctx_str: str) -> Tuple[List[Tuple[str,str]], str, Optional[str]]:
            history = history or []
            if not user_msg or not user_msg.strip():
                return history, "", None
            try:
                context = json.loads(ctx_str or "{}")
            except Exception:
                context = {"mode":"none","prediction":"","top":[],"recommendations":""}

            ctx_mode = context.get("mode","none")
            ctx_pred = context.get("prediction","")
            top_rows = context.get("top", [])
            top_str  = ", ".join([f"{r.get('Class','?')}({r.get('Prob','?')})" for r in top_rows[:3]]) if isinstance(top_rows,list) else "n/a"

            system = (
                "You are a helpful agronomy assistant. Be concise and practical.\n"
                f"Latest context → mode: {ctx_mode}; prediction: {ctx_pred}; top: {top_str}."
            )
            try:
                payload = {"model": cfg["ollama_model"],
                           "messages": [{"role":"system","content":system},
                                        {"role":"user","content":user_msg}],
                           "stream": False}
                r = requests.post("http://localhost:11434/api/chat", json=payload, timeout=cfg["llm_timeout"])
                if r.status_code == 200:
                    reply = (r.json().get("message",{}) or {}).get("content","").strip()
                else:
                    reply = f"(LLM status {r.status_code})"
            except Exception:
                reply = "(LLM unavailable)"

            history = history + [(user_msg, reply)]
            tts_path = None
            if GTTS_OK and reply and not reply.startswith("("):
                tts_path = tempfile.mktemp(suffix=".mp3")
                gTTS(reply).save(tts_path)
            return history, "", tts_path

        chat_send_text.click(
            lambda hist, text, ctx: _chat_core(hist, text, ctx),
            inputs=[chat_box, chat_input, ctx_json],
            outputs=[chat_box, chat_input, chat_tts]
        )

        def chat_voice_fn(hist, voice_path, ctx):
            if not voice_path:
                return hist, "", None
            transcript = whisper_transcribe(voice_path, cfg["whisper_size"])
            if transcript.startswith("(") and "error" in transcript.lower():
                return (hist or []) + [("","(voice error)")], "", None
            return _chat_core(hist, transcript, ctx)

        chat_send_voice.click(
            chat_voice_fn,
            inputs=[chat_box, chat_voice, ctx_json],
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
    ap.add_argument("--ollama_model", default="llama3")
    ap.add_argument("--whisper_size", default="base", help="tiny|base|small|medium|large")
    ap.add_argument("--port", type=int, default=7861)
    ap.add_argument("--llm_timeout", type=int, default=10, help="seconds to wait for Ollama before returning UI")
    args = ap.parse_args()

    build_interface(Path(args.tabular_model_dir), Path(args.vision_model_dir), Path(args.images_dir),
                    args.ollama_model, args.port, args.whisper_size, args.llm_timeout)

if __name__ == "__main__":
    main()
