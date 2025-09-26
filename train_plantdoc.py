#!/usr/bin/env python3
# train_plantdoc_lit.py
"""
Train PlantDoc using literature-backed backbones via timm, keeping repo splits.

Defaults (changeable via --models):
  - tf_efficientnet_b4_ns
  - efficientnet_b3
  - vit_base_patch16_224

Usage:
  python train_plantdoc_lit.py \
    --data_dir ./PlantDoc-Dataset --out_dir ./plantDoc-Output \
    --use_repo_splits --epochs 20 --img_size 384 --batch_size 32
"""

import argparse, json, random, shutil
from pathlib import Path
from typing import List
import numpy as np
import torch, torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform

# -------------------- Utils --------------------
def set_seed(s=42):
    random.seed(s); np.random.seed(s)
    torch.manual_seed(s); torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def plot_cm(cm, names, out_png):
    import numpy as np
    fig, ax = plt.subplots(figsize=(10,8))
    im = ax.imshow(cm, cmap="Blues")
    ax.figure.colorbar(im, ax=ax)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_xticks(np.arange(len(names))); ax.set_yticks(np.arange(len(names)))
    ax.set_xticklabels(names, rotation=90); ax.set_yticklabels(names)
    plt.tight_layout(); plt.savefig(out_png, dpi=200); plt.close(fig)

def stratified_val_from_train_indices(labels: List[int], val_ratio=0.1, seed=42):
    idx = np.arange(len(labels))
    sss = StratifiedShuffleSplit(n_splits=1, test_size=val_ratio, random_state=seed)
    tr_idx, va_idx = next(sss.split(idx, labels))
    return tr_idx, va_idx

def list_classes(root: Path) -> List[str]:
    # class = directory name (ignore files/hidden dirs)
    return sorted([d.name for d in root.iterdir() if d.is_dir() and not d.name.startswith('.')])

class FixedClassImageFolder(datasets.ImageFolder):
    """
    ImageFolder that uses a pre-defined, fixed class list (order matters).
    Ensures train/val/test share identical class_to_idx mappings.
    """
    def __init__(self, root: str | Path, classes: List[str], transform=None):
        self.fixed_classes = classes
        super().__init__(root=str(root), transform=transform)

    def find_classes(self, directory: str):
        classes = self.fixed_classes
        class_to_idx = {c: i for i, c in enumerate(classes)}
        return classes, class_to_idx

# -------------------- Model-size helpers --------------------
def model_requires_img_size_arg(model_name: str) -> bool:
    """Return True for ViT/DeiT-like models that need img_size at init."""
    name = model_name.lower()
    return any(k in name for k in [
        "vit_", "deit", "tnt_", "levit", "beit", "eva", "flexivit"
    ])

# -------------------- Transforms --------------------
def build_transforms_for_model(model_name: str, img_size: int):
    """
    Create a temporary model to fetch the right config, with img_size when needed
    (ViT/DeiT) so transforms match the real model.
    """
    if model_requires_img_size_arg(model_name):
        tmp = timm.create_model(model_name, pretrained=True, num_classes=0, img_size=img_size)
        cfg = resolve_data_config({'img_size': img_size}, model=tmp)
    else:
        tmp = timm.create_model(model_name, pretrained=True, num_classes=0)
        cfg = resolve_data_config({'img_size': img_size}, model=tmp)  # honor requested resize/crop

    train_tf = create_transform(
        input_size=cfg['input_size'],
        interpolation=cfg['interpolation'],
        mean=cfg['mean'], std=cfg['std'],
        crop_pct=cfg.get('crop_pct', None),
        is_training=True, auto_augment='rand-m9-mstd0.5'
    )
    eval_tf  = create_transform(
        input_size=cfg['input_size'],
        interpolation=cfg['interpolation'],
        mean=cfg['mean'], std=cfg['std'],
        crop_pct=cfg.get('crop_pct', None),
        is_training=False
    )
    del tmp
    return train_tf, eval_tf

# -------------------- Data --------------------
def make_loaders(data_dir: Path, seed: int, val_ratio=0.1):
    root_train, root_test = data_dir/"train", data_dir/"test"
    if not root_train.exists() or not root_test.exists():
        raise SystemExit(f"Expected {root_train} and {root_test} (use --use_repo_splits).")

    train_classes = list_classes(root_train)
    test_classes  = list_classes(root_test)

    if train_classes != test_classes:
        train_only = sorted(set(train_classes) - set(test_classes))
        test_only  = sorted(set(test_classes)  - set(train_classes))
        if train_only:
            print("⚠️ Classes only in TRAIN (will be dropped):", train_only)
        if test_only:
            print("⚠️ Classes only in TEST (will be dropped):", test_only)
        common = sorted(set(train_classes) & set(test_classes))
        if not common:
            raise SystemExit("No common classes between train/ and test/. Fix folder names.")
        classes_aligned = common
    else:
        classes_aligned = train_classes  # identical already

    # placeholder datasets (no transforms yet) to get samples/labels and fixed mapping
    ds_train_eval = FixedClassImageFolder(root_train, classes_aligned)
    ds_test_eval  = FixedClassImageFolder(root_test,  classes_aligned)

    labels_train = [ds_train_eval.samples[i][1] for i in range(len(ds_train_eval.samples))]
    tr_idx, va_idx = stratified_val_from_train_indices(labels_train, val_ratio=val_ratio, seed=seed)

    return ds_train_eval, ds_test_eval, tr_idx, va_idx, classes_aligned

# -------------------- Train/Eval --------------------
def train_eval_one(model_name, ds_train_eval, ds_test_eval, tr_idx, va_idx, class_names, out_dir, args):
    # Build transforms for THIS model+img_size
    train_tf, eval_tf = build_transforms_for_model(model_name, args.img_size)

    # Wrap datasets with transforms, preserve the same class mapping
    ds_train_tf = FixedClassImageFolder(ds_train_eval.root, class_names, transform=train_tf)
    ds_val_eval = FixedClassImageFolder(ds_train_eval.root, class_names, transform=eval_tf)
    ds_test     = FixedClassImageFolder(ds_test_eval.root,  class_names, transform=eval_tf)

    train_ds = Subset(ds_train_tf, tr_idx)
    val_ds   = Subset(ds_val_eval, va_idx)

    def loader(ds, shuffle):
        return DataLoader(ds, batch_size=args.batch_size, shuffle=shuffle,
                          num_workers=args.workers, pin_memory=True)

    train_loader = loader(train_ds, True)
    val_loader   = loader(val_ds, False)
    test_loader  = loader(ds_test, False)

    out_arch = out_dir / model_name
    out_arch.mkdir(parents=True, exist_ok=True)
    (out_arch/"label_map.json").write_text(json.dumps({i:c for i,c in enumerate(class_names)}, indent=2))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create model with img_size only when the backbone needs it
    if model_requires_img_size_arg(model_name):
        model = timm.create_model(model_name, pretrained=True,
                                  num_classes=len(class_names), img_size=args.img_size).to(device)
    else:
        model = timm.create_model(model_name, pretrained=True,
                                  num_classes=len(class_names)).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched     = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler    = torch.amp.GradScaler(device.type)  # e.g., 'cuda' if available

    best_val, best_path, wait = 0.0, out_arch/"best_model.pt", 0
    for ep in range(1, args.epochs+1):
        # ------- Train -------
        model.train(); tr_loss=tr_acc=0.0; n=0
        for x,y in train_loader:
            x,y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type):
                logits = model(x); loss = criterion(logits,y)
            scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
            pred = logits.argmax(1)
            tr_acc  += (pred==y).sum().item()
            tr_loss += loss.item()*x.size(0)
            n += x.size(0)
        tr_loss/=n; tr_acc/=n

        # ------- Val -------
        model.eval(); va_loss=va_acc=0.0; n=0
        with torch.no_grad(), torch.amp.autocast(device_type=device.type):
            for x,y in val_loader:
                x,y = x.to(device), y.to(device)
                logits = model(x); loss = criterion(logits,y)
                pred = logits.argmax(1)
                va_acc  += (pred==y).sum().item()
                va_loss += loss.item()*x.size(0)
                n += x.size(0)
        va_loss/=n; va_acc/=n; sched.step()
        print(f"[{model_name}][{ep:03d}] train {tr_loss:.4f}/{tr_acc:.4f}  val {va_loss:.4f}/{va_acc:.4f}")

        if va_acc>best_val:
            best_val=va_acc; torch.save(model.state_dict(), best_path); wait=0
        else:
            wait+=1
            if wait>=args.patience:
                print(f"[{model_name}] early stop"); break

    # ------- Test -------
    state = torch.load(best_path, map_location=device); model.load_state_dict(state)
    y_true, y_pred = [], []
    test_loss=correct=n=0
    with torch.no_grad(), torch.amp.autocast(device_type=device.type):
        for x,y in test_loader:
            x,y = x.to(device), y.to(device)
            logits = model(x); loss = criterion(logits,y)
            pred = logits.argmax(1)
            test_loss += loss.item()*x.size(0)
            correct   += (pred==y).sum().item()
            n += x.size(0)
            y_true.extend(y.cpu().tolist()); y_pred.extend(pred.cpu().tolist())
    te_loss=test_loss/n; te_acc=correct/n
    print(f"[{model_name}][TEST] {te_loss:.4f}/{te_acc:.4f}")

    # ------- Reports -------
    rep = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
    (out_arch/"metrics.json").write_text(json.dumps(
        {"val_best_acc":float(best_val),"test_loss":float(te_loss),
         "test_acc":float(te_acc),"report":rep}, indent=2))
    cm = confusion_matrix(y_true, y_pred); plot_cm(cm, class_names, out_arch/"confusion_matrix.png")
    (out_arch/"vision_meta.json").write_text(json.dumps(
        {"arch":model_name,"img_size":args.img_size,"num_classes":len(class_names)}, indent=2))

    return {"arch": model_name, "val_best_acc": float(best_val),
            "test_acc": float(te_acc), "path": str(best_path)}

# -------------------- Main --------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--models", nargs="+",
        default=["tf_efficientnet_b4_ns","efficientnet_b3","vit_base_patch16_224"])
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--img_size", type=int, default=384,
                    help="Target resize/crop. If ViT/DeiT is used, must be divisible by patch size (usually 16).")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--use_repo_splits", action="store_true", help="Expect data_dir/train and data_dir/test")
    ap.add_argument("--val_ratio", type=float, default=0.1)
    args = ap.parse_args()

    if not args.use_repo_splits:
        raise SystemExit("For PlantDoc repo layout, pass --use_repo_splits (train/test only).")

    # ViT sanity: ensure size divisible by patch size (16 for vit_base_patch16_*)
    if any(model_requires_img_size_arg(m) for m in args.models):
        if args.img_size % 16 != 0:
            raise SystemExit(f"--img_size must be divisible by 16 for ViT/DeiT; got {args.img_size}")

    set_seed(args.seed)
    data_dir, out_dir = Path(args.data_dir), Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    ds_train_eval, ds_test_eval, tr_idx, va_idx, classes = make_loaders(
        data_dir, args.seed, args.val_ratio
    )

    results=[]
    for m in args.models:
        results.append(train_eval_one(m, ds_train_eval, ds_test_eval, tr_idx, va_idx, classes, out_dir, args))

    results=sorted(results, key=lambda r: (r["test_acc"], r["val_best_acc"]), reverse=True)
    (out_dir/"leaderboard.json").write_text(json.dumps({"results":results}, indent=2))
    (out_dir/"best_model_name.txt").write_text(results[0]["arch"])
    best_dst = out_dir/"best_model.pt"
    if best_dst.exists(): best_dst.unlink()
    shutil.copy2(results[0]["path"], best_dst)
    print("\nBest:", results[0])
    print("Saved:", best_dst.resolve())

if __name__ == "__main__":
    main()
