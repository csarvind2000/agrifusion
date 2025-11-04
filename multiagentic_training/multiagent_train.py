#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
multiagent_train.py — AgriFusion Multi-Agent Training (Tabular + PlantDoc)
===========================================================================
Adds:
  • SearchAgent (Optuna) for model selection / tuning
  • SHAP explainability for best tabular model
  • Grad-CAM overlays for best image model
  • MLflow tracking (--mlflow_uri, --experiment)

Usage:
  Tabular CSV:
    python multiagent_train.py --csv data.csv --trials 20 \
      --mlflow_uri sqlite:///mlruns.db --experiment agrifusion

  PlantDoc (image folders per class):
    python multiagent_train.py --plantdoc /path/to/plantdoc --epochs 10 \
      --mlflow_uri sqlite:///mlruns.db --experiment agrifusion
"""

import os, sys, json, time, math, random, shutil
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, List
import cv2
import numpy as np
import torch

# -------------------- Optional deps (handled gracefully) --------------------
try:
    import optuna
except Exception:
    optuna = None

try:
    import shap
except Exception:
    shap = None

try:
    import mlflow
    from mlflow import sklearn as mlflow_sklearn
except Exception:
    mlflow = None

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, StratifiedKFold, KFold
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, r2_score, mean_squared_error
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor

# xgboost optional
try:
    from xgboost import XGBClassifier, XGBRegressor
except Exception:
    XGBClassifier = XGBRegressor = None

# Torch & TorchVision for image / deep tabular
try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, Dataset
    import torchvision
    from torchvision import transforms
except Exception:
    torch = None
    nn = None
    DataLoader = Dataset = None
    torchvision = None
    transforms = None

import argparse
np.random.seed(42)
random.seed(42)

# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def is_classification(y: pd.Series) -> bool:
    # Heuristic: small number of unique values OR dtype is object/bool
    if y.dtype == "O" or y.dtype == "bool":
        return True
    nunique = y.nunique(dropna=True)
    return nunique <= max(20, int(0.05 * len(y)))  # many small classes or clear class labels

def auto_target(df: pd.DataFrame) -> str:
    # Try last column first; otherwise pick the column with the fewest unique values
    last = df.columns[-1]
    if df[last].nunique() <= max(20, int(0.05 * len(df))) or df[last].dtype == "O":
        return last
    # else pick the categorical-ish column
    counts = df.nunique()
    return counts.idxmin()

def split_features(df: pd.DataFrame, target: str) -> Tuple[List[str], List[str]]:
    cat_cols = [c for c in df.columns if c != target and (df[c].dtype == "O" or str(df[c].dtype).startswith("category"))]
    num_cols = [c for c in df.columns if c != target and c not in cat_cols]
    return num_cols, cat_cols

def make_preprocessor(num_cols, cat_cols):
    num_tf = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])
    cat_tf = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                       ("oh", OneHotEncoder(handle_unknown="ignore", sparse_output=False))])
    return ColumnTransformer([("num", num_tf, num_cols),
                              ("cat", cat_tf, cat_cols)])

def metric_bundle(task: str, y_true, y_pred, proba=None) -> Dict[str, float]:
    out = {}
    if task == "classification":
        out["acc"] = accuracy_score(y_true, y_pred)
        out["f1"] = f1_score(y_true, y_pred, average="weighted")
        if proba is not None and len(np.unique(y_true)) == 2:
            try:
                out["auc"] = roc_auc_score(y_true, proba[:, 1])
            except Exception:
                pass
    else:
        out["r2"] = r2_score(y_true, y_pred)
        out["rmse"] = math.sqrt(mean_squared_error(y_true, y_pred))
    return out

def best_key_for(task: str):
    return "f1" if task == "classification" else "r2"

# ---------------------------------------------------------------------------
# SearchAgent (Optuna)
# ---------------------------------------------------------------------------

class SearchAgent:
    def __init__(self, task: str, preprocessor, trials: int = 15, n_splits: int = 5, random_state: int = 42):
        self.task = task
        self.preprocessor = preprocessor
        self.trials = trials
        self.n_splits = n_splits
        self.random_state = random_state
        if optuna is None:
            print("⚠️ Optuna not installed; SearchAgent will run baseline only.")

    def _candidate_models(self):
        models = []
        if self.task == "classification":
            models.append(("logreg", LogisticRegression(max_iter=1000)))
            models.append(("rf", RandomForestClassifier()))
            if XGBClassifier is not None:
                models.append(("xgb", XGBClassifier(tree_method="hist", eval_metric="logloss")))
            models.append(("mlp", MLPClassifier(max_iter=300)))
        else:
            models.append(("ridge", Ridge()))
            models.append(("rf", RandomForestRegressor()))
            if XGBRegressor is not None:
                models.append(("xgb", XGBRegressor(tree_method="hist")))
            models.append(("mlp", MLPRegressor(max_iter=400)))
        return models

    def _objective(self, trial, X, y):
        name= trial.suggest_categorical("model", [n for n,_ in self._candidate_models()])
        # basic hyperparams
        if name == "logreg":
            C = trial.suggest_float("C", 1e-3, 10.0, log=True)
            model = LogisticRegression(C=C, max_iter=1000)
        elif name == "ridge":
            alpha = trial.suggest_float("alpha", 1e-3, 10.0, log=True)
            model = Ridge(alpha=alpha)
        elif name == "rf":
            n_estimators = trial.suggest_int("n_estimators", 100, 800)
            max_depth = trial.suggest_int("max_depth", 3, 20)
            if self.task == "classification":
                model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=self.random_state)
            else:
                model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=self.random_state)
        elif name == "xgb" and (XGBClassifier is not None or XGBRegressor is not None):
            lr = trial.suggest_float("learning_rate", 1e-3, 0.3, log=True)
            depth = trial.suggest_int("max_depth", 3, 10)
            n_estimators = trial.suggest_int("n_estimators", 200, 1000)
            reg = trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True)
            if self.task == "classification":
                model = XGBClassifier(learning_rate=lr, max_depth=depth, n_estimators=n_estimators,
                                      reg_lambda=reg, tree_method="hist", eval_metric="logloss",
                                      random_state=self.random_state)
            else:
                model = XGBRegressor(learning_rate=lr, max_depth=depth, n_estimators=n_estimators,
                                     reg_lambda=reg, tree_method="hist", random_state=self.random_state)
        elif name == "mlp":
            hidden = trial.suggest_categorical("hidden", [(128,), (256,), (128,64)])
            alpha = trial.suggest_float("alpha", 1e-5, 1e-2, log=True)
            if self.task == "classification":
                model = MLPClassifier(hidden_layer_sizes=hidden, alpha=alpha, max_iter=400, random_state=self.random_state)
            else:
                model = MLPRegressor(hidden_layer_sizes=hidden, alpha=alpha, max_iter=600, random_state=self.random_state)
        else:
            # Fallback
            model = LogisticRegression(max_iter=500) if self.task == "classification" else Ridge()

        pipe = Pipeline([("pre", self.preprocessor), ("model", model)])
        # CV
        if self.task == "classification":
            cv = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)
            key = "f1"
        else:
            cv = KFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)
            key = "r2"

        scores = []
        for tr, va in cv.split(X, y):
            pipe.fit(X.iloc[tr], y.iloc[tr])
            yhat = pipe.predict(X.iloc[va])
            if self.task == "classification":
                scores.append(f1_score(y.iloc[va], yhat, average="weighted"))
            else:
                scores.append(r2_score(y.iloc[va], yhat))
        return float(np.mean(scores))

    def search(self, X: pd.DataFrame, y: pd.Series) -> Tuple[Pipeline, Dict[str, Any], float]:
        if optuna is None:
            # Return a strong baseline without tuning
            model = RandomForestClassifier() if self.task == "classification" else RandomForestRegressor()
            pipe = Pipeline([("pre", self.preprocessor), ("model", model)])
            pipe.fit(X, y)
            return pipe, {"model": "rf_baseline"}, 0.0

        study = optuna.create_study(direction="maximize")
        study.optimize(lambda t: self._objective(t, X, y), n_trials=self.trials, show_progress_bar=False)
        best_params = study.best_trial.params
        # rebuild best model
        name = best_params.get("model", "rf")
        # map again
        trial = optuna.trial.FixedTrial(best_params)
        _ = self._objective(trial, X.iloc[:5], y.iloc[:5])  # to construct the model structure
        # The previous call built and fit inside CV; rebuild fresh with same params:
        model = None
        if name == "logreg":
            model = LogisticRegression(C=best_params["C"], max_iter=1000)
        elif name == "ridge":
            model = Ridge(alpha=best_params["alpha"])
        elif name == "rf":
            model = RandomForestClassifier(n_estimators=best_params["n_estimators"],
                                           max_depth=best_params["max_depth"],
                                           random_state=self.random_state) if self.task=="classification" \
                    else RandomForestRegressor(n_estimators=best_params["n_estimators"],
                                               max_depth=best_params["max_depth"],
                                               random_state=self.random_state)
        elif name == "xgb" and (XGBClassifier is not None or XGBRegressor is not None):
            if self.task == "classification":
                model = XGBClassifier(learning_rate=best_params["learning_rate"],
                                      max_depth=best_params["max_depth"],
                                      n_estimators=best_params["n_estimators"],
                                      reg_lambda=best_params["reg_lambda"],
                                      tree_method="hist", eval_metric="logloss",
                                      random_state=self.random_state)
            else:
                model = XGBRegressor(learning_rate=best_params["learning_rate"],
                                     max_depth=best_params["max_depth"],
                                     n_estimators=best_params["n_estimators"],
                                     reg_lambda=best_params["reg_lambda"],
                                     tree_method="hist", random_state=self.random_state)
        elif name == "mlp":
            if self.task == "classification":
                model = MLPClassifier(hidden_layer_sizes=best_params["hidden"], alpha=best_params["alpha"],
                                      max_iter=400, random_state=self.random_state)
            else:
                model = MLPRegressor(hidden_layer_sizes=best_params["hidden"], alpha=best_params["alpha"],
                                     max_iter=600, random_state=self.random_state)
        else:
            model = RandomForestClassifier() if self.task == "classification" else RandomForestRegressor()

        pipe = Pipeline([("pre", self.preprocessor), ("model", model)])
        pipe.fit(X, y)
        return pipe, best_params, study.best_value

# ---------------------------------------------------------------------------
# SHAP Explainability (Tabular)
# ---------------------------------------------------------------------------

def run_shap_for_tabular(pipeline: Pipeline, X_train: pd.DataFrame, outdir: Path, max_samples: int = 1000):
    if shap is None:
        print("⚠️ SHAP not installed; skipping SHAP plots.")
        return None
    ensure_dir(outdir)
    # Get transformed matrix & inner model
    pre = pipeline.named_steps["pre"]
    model = pipeline.named_steps["model"]
    # Subsample for tractability
    Xs = X_train.sample(min(len(X_train), max_samples), random_state=42)
    Xp = pre.fit_transform(Xs.copy())  # NOTE: uses its own fitted params (ok for explainability thumbnail)

    # Pick explainer
    try:
        explainer = shap.Explainer(model)
        sv = explainer(Xp)
    except Exception:
        # kernel fallback
        explainer = shap.KernelExplainer(model.predict, Xp[:50])
        sv = explainer.shap_values(Xp[:200], nsamples=100)

    # Plots
    try:
        shap.summary_plot(sv, show=False)
        plt.tight_layout()
        plt.savefig(outdir / "shap_summary.png", dpi=200)
        plt.close()
    except Exception:
        pass

    try:
        shap.summary_plot(sv, plot_type="bar", show=False)
        plt.tight_layout()
        plt.savefig(outdir / "shap_bar.png", dpi=200)
        plt.close()
    except Exception:
        pass

# ---------------------------------------------------------------------------
# PlantDoc Dataset + Grad-CAM
# ---------------------------------------------------------------------------

class PlantDocDataset(Dataset):
    def __init__(self, root: Path, train: bool = True, img_size: int = 224):
        self.root = Path(root)
        split = "train" if train else "val"
        # if explicit split exists, use it; else perform a simple split on the fly
        self.paths = []
        self.labels = []
        classes = sorted([d.name for d in self.root.iterdir() if d.is_dir()])
        self.class_to_idx = {c: i for i, c in enumerate(classes)}
        for cls in classes:
            files = list((self.root / cls).glob("**/*.jpg")) + list((self.root / cls).glob("**/*.png"))
            random.Random(42).shuffle(files)
            cut = int(0.8 * len(files))
            sel = files[:cut] if train else files[cut:]
            for p in sel:
                self.paths.append(p)
                self.labels.append(self.class_to_idx[cls])
        self.tf = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
        ])

    def __len__(self): return len(self.paths)
    def __getitem__(self, idx):
        from PIL import Image
        img = Image.open(self.paths[idx]).convert("RGB")
        x = self.tf(img)
        y = self.labels[idx]
        return x, y, str(self.paths[idx])

def build_image_model(name: str, num_classes: int):
    if torchvision is None:
        raise RuntimeError("torchvision not installed")
    if name == "resnet18":
        model = torchvision.models.resnet18(weights="IMAGENET1K_V1")
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
        target_layer = model.layer4[-1].conv2  # for Grad-CAM
    elif name == "efficientnet_b0":
        model = torchvision.models.efficientnet_b0(weights="IMAGENET1K_V1")
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, num_classes)
        target_layer = model.features[-1][0]
    else:
        model = torchvision.models.mobilenet_v3_small(weights="IMAGENET1K_V1")
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, num_classes)
        target_layer = model.features[-1][0]
    return model, target_layer

def train_image_model(root: Path, outdir: Path, epochs=5, batch=32, lr=1e-3, model_name="resnet18",
                      device=None, mlflow_ctx: Optional[Dict[str,Any]]=None):
    ensure_dir(outdir)
    if torch is None:
        raise RuntimeError("PyTorch not installed.")
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    ds_tr = PlantDocDataset(root, train=True)
    ds_va = PlantDocDataset(root, train=False)
    dl_tr = DataLoader(ds_tr, batch_size=batch, shuffle=True, num_workers=2)
    dl_va = DataLoader(ds_va, batch_size=batch, shuffle=False, num_workers=2)
    num_classes = len(ds_tr.class_to_idx)

    model, target_layer = build_image_model(model_name, num_classes)
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    ce = nn.CrossEntropyLoss()

    best_acc, best_path = 0.0, outdir / f"{model_name}_best.pt"
    for ep in range(1, epochs+1):
        model.train()
        tr_loss, tr_correct, n = 0.0, 0, 0
        for xb, yb, _ in dl_tr:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            logits = model(xb)
            loss = ce(logits, yb)
            loss.backward()
            opt.step()
            tr_loss += loss.item() * xb.size(0)
            tr_correct += (logits.argmax(1) == yb).sum().item()
            n += xb.size(0)
        tr_acc = tr_correct / max(1, n)

        # val
        model.eval()
        va_correct, m = 0, 0
        with torch.no_grad():
            for xb, yb, _ in dl_va:
                xb, yb = xb.to(device), yb.to(device)
                logits = model(xb)
                va_correct += (logits.argmax(1) == yb).sum().item()
                m += xb.size(0)
        va_acc = va_correct / max(1, m)

        print(f"[{model_name}] epoch {ep}/{epochs} | train_acc={tr_acc:.3f} val_acc={va_acc:.3f}")
        if mlflow is not None and mlflow_ctx is not None:
            mlflow.log_metrics({"train_acc": tr_acc, "val_acc": va_acc}, step=ep)
        if va_acc > best_acc:
            best_acc = va_acc
            torch.save(model.state_dict(), best_path)

    # load best
    model.load_state_dict(torch.load(best_path, map_location=device))
    return model, target_layer, best_acc, ds_va

def grad_cam_overlays(model, dataloader, device, outdir):
    os.makedirs(outdir, exist_ok=True)
    model.eval()

    from torchvision.transforms.functional import to_pil_image
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
    from pytorch_grad_cam.utils.image import show_cam_on_image

    target_layers = [model.features[-1]]  # For MobileNetV3/ResNet etc.
    cam = GradCAM(model=model, target_layers=target_layers, use_cuda=torch.cuda.is_available())

    for i, (img, label) in enumerate(dataloader):
        img = img.to(device)
        targets = [ClassifierOutputTarget(label.item())]

        grayscale_cam = cam(input_tensor=img, targets=targets)[0]  # (H', W')

        # Convert tensor to numpy image
        img_np = img.cpu().permute(1, 2, 0).numpy()
        img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min())

        # Resize heatmap to match input image size
        heatmap_resized = cv2.resize(grayscale_cam, (img_np.shape[1], img_np.shape[0]))
        heatmap_rgb = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
        heatmap_rgb = cv2.cvtColor(heatmap_rgb, cv2.COLOR_BGR2RGB) / 255.0

        # Blend overlay
        overlay = (0.5 * img_np + 0.5 * heatmap_rgb)
        overlay = np.clip(overlay * 255, 0, 255).astype(np.uint8)

        # Save
        out_path = os.path.join(outdir, f"gradcam_{i:03d}.png")
        cv2.imwrite(out_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

    print(f"✅ Grad-CAM overlays saved to {outdir}")

# ---------------------------------------------------------------------------
# Tabular Training Orchestrator
# ---------------------------------------------------------------------------

def train_tabular(csv_path: Path, target: Optional[str], trials: int, outdir: Path,
                  mlflow_uri: Optional[str], experiment: Optional[str]):
    ensure_dir(outdir)
    df = pd.read_csv(csv_path)
    if target is None:
        target = auto_target(df)
        print(f"🧭 Auto-detected target column: {target}")
    y = df[target]
    X = df.drop(columns=[target])

    task = "classification" if is_classification(y) else "regression"
    print(f"🔎 Detected task: {task}")

    num_cols, cat_cols = split_features(df, target)
    pre = make_preprocessor(num_cols, cat_cols)
    searcher = SearchAgent(task, pre, trials=trials)

    # MLflow setup
    if mlflow is not None and mlflow_uri and experiment:
        mlflow.set_tracking_uri(mlflow_uri)
        mlflow.set_experiment(experiment)

    with (mlflow.start_run(run_name=f"tabular_{csv_path.stem}") if mlflow and mlflow_uri and experiment else nullcontext()):
        best_pipe, best_params, best_cv = searcher.search(X, y)
        # holdout evaluation
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y if task=="classification" else None)
        best_pipe.fit(Xtr, ytr)
        yhat = best_pipe.predict(Xte)
        proba = None
        if task == "classification":
            try:
                proba = best_pipe.predict_proba(Xte)
            except Exception:
                proba = None
        metrics = metric_bundle(task, yte, yhat, proba)
        print("✅ Holdout metrics:", metrics)

        # Save model
        import joblib
        model_path = outdir / "best_tabular_model.joblib"
        joblib.dump(best_pipe, model_path)
        with open(outdir / "best_params.json", "w") as f:
            json.dump(best_params, f, indent=2)

        # SHAP explainability
        run_shap_for_tabular(best_pipe, Xtr, outdir=outdir / "shap")

        # Log to MLflow
        if mlflow is not None and mlflow_uri and experiment:
            mlflow.log_params(best_params)
            mlflow.log_metrics(metrics)
            mlflow.log_artifact(model_path)
            shap_dir = outdir / "shap"
            if shap_dir.exists():
                for p in shap_dir.glob("*.png"):
                    mlflow.log_artifact(p)

    # return path summary
    return {"model_path": str(outdir / "best_tabular_model.joblib"), "metrics": metrics, "task": task}

# ---------------------------------------------------------------------------
# Image Training Orchestrator (PlantDoc)
# ---------------------------------------------------------------------------

from contextlib import contextmanager
@contextmanager
def nullcontext():
    yield

def train_plantdoc(data_root: Path, outdir: Path, epochs: int, trials: int,
                   mlflow_uri: Optional[str], experiment: Optional[str]):
    ensure_dir(outdir)
    # Candidate models + simple search over LR
    model_names = ["resnet18", "efficientnet_b0", "mobilenet_v3_small"]
    lrs = [5e-4, 1e-3, 2e-3]

    if mlflow is not None and mlflow_uri and experiment:
        mlflow.set_tracking_uri(mlflow_uri)
        mlflow.set_experiment(experiment)

    best = {"acc": -1.0, "name": None, "lr": None}
    best_bundle = None

    for name in model_names:
        for lr in lrs:
            rn = f"plantdoc_{name}_lr{lr}"
            with (mlflow.start_run(run_name=rn) if mlflow and mlflow_uri and experiment else nullcontext()):
                model, target_layer, val_acc, ds_val = train_image_model(
                    data_root, outdir=outdir / rn, epochs=epochs, lr=lr, model_name=name,
                    mlflow_ctx={"name": rn} if mlflow else None
                )
                if mlflow is not None and mlflow_uri and experiment:
                    mlflow.log_params({"model": name, "lr": lr})
                    mlflow.log_metrics({"val_acc": val_acc})

                if val_acc > best["acc"]:
                    best.update({"acc": val_acc, "name": name, "lr": lr})
                    best_bundle = (model, target_layer, ds_val)

    print(f"🏆 Best image model: {best}")

    # Grad-CAM overlays for the winner
    if best_bundle:
        gc_dir = outdir / "gradcam_samples"
        grad_cam_overlays(*best_bundle, outdir=gc_dir)
        if mlflow is not None and mlflow_uri and experiment and gc_dir.exists():
            for p in gc_dir.glob("*.png"):
                mlflow.log_artifact(p)

    # Save simple manifest
    with open(outdir / "best_image_model.json", "w") as f:
        json.dump(best, f, indent=2)

    return best

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    ap = argparse.ArgumentParser(description="AgriFusion Multi-Agent Training")
    ap.add_argument("--csv", type=str, default=None, help="Path to input CSV (tabular).")
    ap.add_argument("--target", type=str, default=None, help="Optional target column; auto-detected if omitted.")
    ap.add_argument("--plantdoc", type=str, default=None, help="Path to PlantDoc data (classes as subfolders).")
    ap.add_argument("--out", type=str, default="outputs_train", help="Output directory.")
    ap.add_argument("--trials", type=int, default=15, help="Optuna trials for tabular.")
    ap.add_argument("--epochs", type=int, default=6, help="Epochs for image training.")
    ap.add_argument("--mlflow_uri", type=str, default=None, help="e.g., sqlite:///mlruns.db")
    ap.add_argument("--experiment", type=str, default=None, help="MLflow experiment name.")
    return ap.parse_args()

def main():
    args = parse_args()
    outdir = Path(args.out)
    ensure_dir(outdir)

    summary = {}

    if args.csv:
        print("==== TABULAR PIPELINE ====")
        tdir = outdir / "tabular"
        ensure_dir(tdir)
        tab = train_tabular(Path(args.csv), args.target, args.trials, tdir, args.mlflow_uri, args.experiment)
        summary["tabular"] = tab

    if args.plantdoc:
        if torch is None or torchvision is None:
            print("❌ Torch/torchvision required for PlantDoc pipeline.")
        else:
            print("==== PLANTDOC PIPELINE ====")
            pdir = outdir / "plantdoc"
            ensure_dir(pdir)
            img = train_plantdoc(Path(args.plantdoc), pdir, args.epochs, args.trials, args.mlflow_uri, args.experiment)
            summary["plantdoc"] = img

    with open(outdir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("✅ Done. Summary saved to", outdir / "summary.json")

if __name__ == "__main__":
    main()
