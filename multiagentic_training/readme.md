# 🌾 Multi-Agentic Training System — Agentic Workflow Overview

This section explains **how the agents interact and collaborate** in the AgriFusion Multi-Agentic Training pipeline.
Each agent is autonomous yet cooperative, contributing a specialized task in the overall AI workflow.

---

## 🧠 1. Core Principle — Agentic Collaboration

The pipeline uses **modular agents** that communicate via a shared context.
Each agent performs a distinct function such as data preparation, model optimization, training, explainability, or experiment logging.

Agents are orchestrated sequentially or in parallel depending on the task (e.g., tabular vs. image data).

---

## 🧩 2. Agentic Architecture

| Agent                      | Role                                                 | Key Tools                     | Output                                  |
| -------------------------- | ---------------------------------------------------- | ----------------------------- | --------------------------------------- |
| **DataAgent**              | Loads dataset, detects target variable and task type | pandas, scikit-learn          | Features (`X`), Labels (`y`), Task type |
| **SearchAgent (Optuna)**   | Performs hyperparameter optimization                 | Optuna, scikit-learn, PyTorch | Best model configuration                |
| **TrainingAgent**          | Trains models (ML, DL, Transformer)                  | PyTorch, sklearn              | Trained model + metrics                 |
| **ExplainAgent (SHAP)**    | Generates feature importance & interpretation        | SHAP                          | SHAP plots & summary                    |
| **VisionAgent (Grad-CAM)** | Visual explainability for CNNs                       | Grad-CAM, torchvision         | Grad-CAM overlays                       |
| **MlflowAgent**            | Tracks all experiments, parameters & artifacts       | MLflow                        | Logs, visual dashboards                 |

---

## 🔄 3. Interaction Flow

```
┌──────────────────┐
│ DataAgent        │
│ (detect features)│
└──────┬───────────┘
       │
       ▼
┌──────────────────┐
│ SearchAgent      │
│ (Optuna tuning)  │
└──────┬───────────┘
       │
       ▼
┌──────────────────┐
│ TrainingAgent    │
│ (fit + evaluate) │
└──────┬───────────┘
       │
       ▼
┌──────────────────┐
│ ExplainAgent     │
│ (SHAP/GradCAM)   │
└──────┬───────────┘
       │
       ▼
┌──────────────────┐
│ MlflowAgent      │
│ (log + manage)   │
└──────────────────┘
```

---

## ⚙️ 4. Communication Logic

Agents interact via a **shared `context` dictionary**.
Each agent reads inputs from the context, performs computation, and updates it for the next agent.

**Example:**

```python
context = DataAgent().run({})
context = SearchAgent().run(context)
context = TrainingAgent().run(context)
context = ExplainAgent().run(context)
context = MlflowAgent().run(context)
```

Each agent’s `run()` method:

* Reads relevant keys from `context`
* Performs its computation (e.g., training or SHAP analysis)
* Adds its results back into `context` for downstream agents

---

## 🧩 5. Interaction Example (CSV Training)

### Step 1 – **DataAgent**

```python
context = {
  "X": <features>,
  "y": <target>,
  "task": "classification"
}
```

### Step 2 – **SearchAgent (Optuna)**

* Runs 20–50 hyperparameter trials
* Picks best-performing model

Updates:

```python
context["best_model"] = "RandomForestClassifier"
context["best_params"] = {"n_estimators": 200, "max_depth": 10}
```

### Step 3 – **TrainingAgent**

* Fits model on training data
* Evaluates on holdout validation

Updates:

```python
context["metrics"] = {"accuracy": 0.994, "f1": 0.993}
context["model_path"] = "outputs_train/tabular/rf_best.pkl"
```

### Step 4 – **ExplainAgent (SHAP)**

* Computes feature importance
* Saves SHAP summary and dependence plots

Updates:

```python
context["shap_summary"] = "outputs_train/tabular/shap/shap_summary.png"
```

### Step 5 – **MlflowAgent**

* Logs all parameters, metrics, and artifacts
* Registers experiment under name `agrifusion`

---

## 🧮 6. Vision Workflow (PlantDoc)

For image classification (PlantDoc dataset), VisionAgent replaces SHAPAgent.

**Interaction Flow:**

```
VisionAgent → TrainingAgent → GradCAMAgent → MlflowAgent
```

Each model (ResNet, EfficientNet, MobileNet) is trained separately.
Grad-CAM visualizations highlight attention regions on diseased leaves.

Artifacts saved:

```
outputs_train/plantdoc/
 ├── resnet18_best.pt
 ├── gradcam_samples/
 ├── summary.json
 └── metrics.csv
```

---

## 🧩 7. MLflow Integration

Each agent logs to MLflow through a unified tracking interface:

```bash
--mlflow_uri sqlite:///mlruns.db
--experiment agrifusion
```

Logs include:

* **Parameters:** model type, hyperparameters, learning rate, etc.
* **Metrics:** accuracy, loss, F1-score, etc.
* **Artifacts:** SHAP plots, Grad-CAM overlays, trained models

Launch the UI:

```bash
mlflow ui --backend-store-uri sqlite:///mlruns.db
```

Then open:

```
http://localhost:5000
```

---

## 🧭 8. Multi-Agent Synchronization

Both **tabular** and **vision** pipelines can run concurrently:

* Each registers as a sub-experiment within MLflow
* The **MlflowAgent** synchronizes logging to a shared experiment
* Outputs are consolidated into `summary.json`

---

## 💡 9. Future Enhancements

| Extension                 | Description                                                                   |
| ------------------------- | ----------------------------------------------------------------------------- |
| **LLMAgent**              | Uses an LLM (e.g., Phi-3, Llama 3) to suggest model architectures dynamically |
| **RewardAgent**           | RL-based reward-driven optimization (PPO, DQN)                                |
| **FeedbackAgent**         | Human-in-loop corrections for model bias                                      |
| **LangGraph Integration** | Visual orchestration of agent reasoning flow                                  |
| **AutoDocAgent**          | Generates PDF reports summarizing training outcomes                           |

---

## 📁 10. Output Structure

```
outputs_train/
 ├── tabular/
 │   ├── shap/
 │   ├── best_model.pkl
 │   └── summary.json
 ├── plantdoc/
 │   ├── gradcam_samples/
 │   ├── best_model.pt
 │   └── metrics.csv
 └── combined_summary.json
```

---

## 🧠 11. Summary of Agent Roles

| Agent                            | Trigger          | Output           | Next Agent    |
| -------------------------------- | ---------------- | ---------------- | ------------- |
| **DataAgent**                    | Dataset detected | Features, target | SearchAgent   |
| **SearchAgent**                  | Data available   | Best hyperparams | TrainingAgent |
| **TrainingAgent**                | Params ready     | Model, metrics   | ExplainAgent  |
| **ExplainAgent (SHAP/Grad-CAM)** | Model trained    | Explanations     | MlflowAgent   |
| **MlflowAgent**                  | Context update   | Logged run       | Dashboard     |

---

## 🧩 12. Example Run Summary

```
==== TABULAR PIPELINE ====
🧭 Auto-detected target: Crop
🔎 Detected task: classification
✅ Best model: RandomForestClassifier
✅ Accuracy: 0.9954 | F1: 0.9954
✅ SHAP summary saved: outputs_train/tabular/shap_summary.png
✅ Run logged in MLflow: agrifusion/exp-23
```

---

**Author:** Arvind C.S.
**License:** MIT
**Repository:** [AgriFusion — Multi-Agent AI for Smart Farming](https://github.com/csarvind2000/agrifusion)
