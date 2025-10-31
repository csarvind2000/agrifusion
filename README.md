# 🌿 AgriFusion — An LLM-Powered Agentic Multilingual AI Assistant for Vision, Voice & Smart Farming

**AgriFusion (AgroSense Studio)** is an intelligent, **multimodal–multilingual LLM system** designed to assist farmers, agronomists, and researchers with real-time crop prediction, disease identification, and agronomy recommendations.  
It integrates **tabular soil data**, **plant images**, **text queries**, and **voice interactions** into one unified interface powered by **Large Language Models (LLMs)** and **agentic routing logic**.

---

## 🧠 Key Features

### 🌾 Multimodal Intelligence
- **Tabular Mode:** Predicts suitable crops or fertilizers using trained ML/DL models.  
- **Vision Mode:** Classifies plant diseases using `timm`-based EfficientNet or ViT models.  
- **Text Mode:** Answers agronomy-related questions using LLMs.  
- **Voice Mode:** Listens and responds to spoken queries using **Whisper** + **gTTS**.

### 🗣️ Multilingual Interface
Supports six Indian languages natively:
> **English**, **हिन्दी (Hindi)**, **ಕನ್ನಡ (Kannada)**, **தமிழ் (Tamil)**, **മലയാളം (Malayalam)**, **తెలుగు (Telugu)**

Both the **UI labels** and **LLM-generated responses** adapt to the selected language.

## 🤖 Multi-Agent Architecture (LLM-Powered Workflow)

**AgriFusion** is built on a modular, *agentic reasoning framework* powered by **Autogen** and **Ollama-based LLMs**.  
Each agent specializes in a different input modality (vision, text, tabular, or voice) and collaborates through a controller to produce coherent multilingual responses.

### 🧩 Agent Roles

| Agent | Responsibility | Example Task |
|:------|:---------------|:--------------|
| 🧭 **RouterAgent** | Detects which inputs are active (image, text, voice, or soil data) and decides which agents to activate. | Routes an image upload to the VisionAgent, or a voice query to the VoiceAgent. |
| 🌿 **VisionAgent** | Uses the EfficientNet/ViT model to identify plant species or detect leaf disease. | “Leaf shows symptoms of rust. Possible cause: fungal infection.” |
| 🌱 **TabularAgent** | Predicts the most suitable crop or fertilizer using soil & climate parameters (N, P, K, pH, temperature, humidity, rainfall). | “Recommended crop: maize; soil nutrients balanced for K but low N.” |
| 🎙️ **VoiceAgent** | Transcribes spoken queries using Whisper ASR and forwards the text to the LLM. | Converts: “What fertilizer should I use for rice?” → text query. |
| 💬 **TextAgent** | Handles natural language Q&A in multiple languages using the local LLM (`llama3` via Ollama). | Responds to “How can I prevent tomato blight?” |
| 🧠 **ReasoningAgent** | Synthesizes outputs from all active agents into one unified, contextualized, multilingual recommendation. | Merges crop prediction, disease detection, and user question into a single answer. |
| 🔊 **TTSAgent** | Converts the final LLM answer into speech using **gTTS** for accessible voice feedback. | Speaks out the recommendation in the user’s selected language. |

---

### ⚙️ Agentic Routing Logic

The following workflow defines how AgriFusion dynamically orchestrates reasoning across multiple agents:

Farmer Input
│
▼
🧭 RouterAgent → decides route
├── 🌿 VisionAgent (if image uploaded)
├── 🌱 TabularAgent (if NPK/pH/Temp/Humidity provided)
├── 🎙️ VoiceAgent (if voice recorded)
└── 💬 TextAgent (if text query given)
▼
🧠 ReasoningAgent → merges results, generates multilingual summary
▼
🔊 TTSAgent → produces voice output
▼
🖥️ Gradio Interface → displays text, tables, and audio

| Component | Library |
|------------|----------|
| Agent Coordination | **Autogen** |
| LLM Backbone | **Ollama** (Llama 3 / Mistral / Phi) |
| Speech Recognition | **Whisper** |
| Vision Model | **EfficientNet B3 (timm)** |
| Speech Synthesis | **gTTS** |
| Interface | **Gradio v4** |


### 💬 Interactive LLM Chat
- Integrated **LLM-powered chat assistant** for continuous interaction.  
- Maintains context of predictions (e.g., detected crop/disease) for deeper follow-up questions.  
- Supports **voice-based** chat using Whisper and speech synthesis with gTTS.

### 🎨 Elegant Gradio UI
- Responsive **3-panel layout** for Inputs • Results • Chat.  
- Dynamic label translation when changing languages.  
- Custom CSS with soft green theme for a modern agricultural feel.

---

## 🚀 Features  

- **📊 Soil & Climate Insights** – Input NPK, pH, temperature, humidity, and rainfall for crop suitability predictions.  
- **🌿 Plant Health Diagnosis** – Upload crop/leaf images for disease detection and management advice.  
- **💬 Conversational AI** – Ask questions via text or voice, get clear agronomy recommendations.  
- **🧠 Multimodal AI Reasoning** – Automatically selects the right AI pipeline (soil, image, text, or voice).  
- **🔊 Voice Output** – Results and recommendations can be spoken back to you.  

---

## 📂 Project Structure  

```
AgriFusion/
│
├── crop_multimodal_app.py   # Main Gradio application
├── m2_outputs/              # Tabular crop prediction models
├── plantDoc-Output/         # Vision models (PlantDoc)
├── images/                  # Sample input images
└── README.md
```

---

## 🔑 Model Files  

The trained models are hosted on **Google Drive**.  
👉 [Download Models](https://drive.google.com/drive/folders/1rr0z26Z8zI4fLhgQHw_FpFI0e7pxo3iI?usp=drive_link)  

After downloading:  
- Copy the **`m2_outputs`** and **`plantDoc-Output`** folders into the **project root folder**.  
- Your folder should look like:  

```
AgriFusion/
│
├── crop_multimodal_app.py
├── m2_outputs/
│   ├── best_model.pt
│   ├── label_encoder.joblib
│   └── ...
├── plantDoc-Output/
│   └── efficientnet_b3/
│       ├── best_model.pt
│       ├── vision_meta.json
│       └── ...
```

---

## ⚙️ Local Installation  

1. **Clone this repository**  
   ```bash
   git clone https://github.com/<your-username>/AgriFusion.git
   cd AgriFusion
   ```

2. **Create environment & install dependencies**  
   ```bash
   conda create -n agrifusion python=3.10 -y
   conda activate agrifusion
   pip install -r requirements.txt
   ```

   Example `requirements.txt`:  
   ```
   gradio
   torch
   torchvision
   timm
   joblib
   scikit-learn
   pillow
   requests
   gTTS
   openai-whisper
   ```

---

## ▶️ Running the App Locally  

```bash
python crop_multimodal_app.py     --tabular_model_dir m2_outputs     --vision_model_dir plantDoc-Output/efficientnet_b3     --images_dir images     --ollama_model llama3     --whisper_size base     --port 7861
```
- multiligual code run 
```bash
python crop_multimodal_app_multilang.py       --tabular_model_dir m2_outputs       --vision_model_dir plantDoc-Output/efficientnet_b3       --images_dir images       --ollama_model llama3       --whisper_size base       --port 7863       --default_lang en
```

- multi-agentic code run 
```bash
python agrifusion_multiagent_ui.py
```

- Open [http://localhost:7861](http://localhost:7861) in your browser.  
- Use soil inputs, upload plant images, or ask questions via text/voice.  

---

## 💻 Google Colab Setup  

You can also run AgriFusion on **Google Colab** without local installation.  

1. Open a new **Google Colab Notebook**.  
2. Clone the repository:  
   ```python
   !git clone https://github.com/<your-username>/AgriFusion.git
   %cd AgriFusion
   ```
3. Install dependencies:  
   ```python
   !pip install gradio torch torchvision timm joblib scikit-learn pillow requests gTTS openai-whisper
   ```
4. Download model files from Google Drive and place them inside the project folder:  
   - [Model Drive Link](https://drive.google.com/drive/folders/1rr0z26Z8zI4fLhgQHw_FpFI0e7pxo3iI?usp=drive_link)  
   - Upload them manually via Colab’s file upload, or use `gdown` if shared links are direct-downloadable.  

   Example using `gdown` (if you make direct links):  
   ```python
   !pip install gdown
   !gdown --folder https://drive.google.com/drive/folders/1rr0z26Z8zI4fLhgQHw_FpFI0e7pxo3iI
   ```

5. Run the app in Colab (with public Gradio link):  
   ```python
   !python crop_multimodal_app.py        --tabular_model_dir m2_outputs        --vision_model_dir plantDoc-Output/efficientnet_b3        --images_dir images        --ollama_model llama3        --whisper_size base        --port 7860 --share
   ```

   The `--share` flag will give you a **public Gradio link** to test your app in Colab.  

---

## 🎯 Applications  

- Sustainable crop planning  
- Disease detection & prevention  
- Precision irrigation & fertilizer use  
- AI-powered digital farming assistants  

---

## 📜 License  

This project is released under the **MIT License**.  

---

✨ **AgriFusion — Smarter Farming, Powered by Multimodal AI.**  


