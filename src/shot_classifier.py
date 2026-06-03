

from __future__ import annotations

import os
import shutil
import zipfile
import urllib.request
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from efficientnet_pytorch import EfficientNet




SHOT_CLASSES = [
    "cover_drive", "straight_drive", "on_drive",  "off_drive",
    "flick_shot",  "square_cut",     "pull_shot",  "hook_shot",
    "sweep_shot",  "lofted_shot",
]
NUM_CLASSES  = len(SHOT_CLASSES)   # 10
NUM_FRAMES   = 16                  # frames sampled per clip
FEATURE_DIM  = 1280                # EfficientNet-B0 output dim
GRU_HIDDEN   = 256                 # hidden units per direction
GRU_LAYERS   = 2
DROPOUT      = 0.5

# Paths
PRETRAINED_URL   = (
    "https://github.com/RITIK-12/CricketShotClassification"
    "/releases/download/v1.0/cricket_shot_model.pth"
)
PRETRAINED_PATH  = "models/ritik12_pretrained.pth"   # downloaded source
FINETUNED_PATH   = "models/shot_classifier.pt"       # our fine-tuned output
DATA_ROOT        = "data/CricShot10"


# ── Image transform ───────────────────────────────────────────────────────────

_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])



class TemporalAttention(nn.Module):
    """Scalar importance weight per frame, softmax pooling."""
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.attn = nn.Linear(hidden_dim, 1)

    def forward(self, gru_out: torch.Tensor) -> torch.Tensor:
        # gru_out : (B, T, H)  →  context : (B, H)
        scores  = self.attn(gru_out).squeeze(-1)          # (B, T)
        weights = F.softmax(scores, dim=1).unsqueeze(2)   # (B, T, 1)
        return (gru_out * weights).sum(dim=1)             # (B, H)


class ShotClassifier(nn.Module):
    """EfficientNet-B0  →  BiGRU  →  TemporalAttention  →  Linear(10)"""

    def __init__(self, num_classes: int = NUM_CLASSES):
        super().__init__()

        # CNN backbone — feature extractor only (no EfficientNet head)
        effnet = EfficientNet.from_pretrained("efficientnet-b0")
        self.cnn = nn.Sequential(
            effnet._conv_stem,
            effnet._bn0,
            *effnet._blocks,
            effnet._conv_head,
            effnet._bn1,
            nn.AdaptiveAvgPool2d(1),
        )
        self.cnn_drop = nn.Dropout(p=0.3)

        # Temporal model
        self.gru = nn.GRU(
            input_size=FEATURE_DIM,
            hidden_size=GRU_HIDDEN,
            num_layers=GRU_LAYERS,
            batch_first=True,
            bidirectional=True,
            dropout=DROPOUT if GRU_LAYERS > 1 else 0.0,
        )
        gru_out_dim = GRU_HIDDEN * 2   # bidirectional

        self.attention = TemporalAttention(gru_out_dim)

        # Classifier head — the only part we fine-tune
        self.head = nn.Sequential(
            nn.LayerNorm(gru_out_dim),
            nn.Dropout(DROPOUT),
            nn.Linear(gru_out_dim, 256),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(256, num_classes),
        )

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.cnn(x)       # (B*T, 1280, 1, 1)
        feat = feat.flatten(1)   # (B*T, 1280)
        return self.cnn_drop(feat)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C, H, W = x.shape
        feat    = self.extract_features(x.view(B * T, C, H, W)).view(B, T, -1)
        gru_out, _ = self.gru(feat)
        context = self.attention(gru_out)
        return self.head(context)

   

    def freeze_backbone(self):
        """Freeze CNN + GRU — only head will be trained."""
        for param in self.cnn.parameters():
            param.requires_grad = False
        for param in self.gru.parameters():
            param.requires_grad = False
        for param in self.attention.parameters():
            param.requires_grad = False
        n_frozen = sum(1 for p in self.parameters() if not p.requires_grad)
        n_total  = sum(1 for _ in self.parameters())
        print(f"[freeze] {n_frozen}/{n_total} parameter groups frozen — only head trains")

    def unfreeze_all(self):
        """Unfreeze everything (for a second full fine-tune pass if desired)."""
        for param in self.parameters():
            param.requires_grad = True
        print("[unfreeze] all layers trainable")



def download_pretrained_weights(dest: str = PRETRAINED_PATH) -> str:
    """
    Download the RITIK-12 pretrained checkpoint.

    NOTE: The URL below points to the GitHub releases page.
    If the direct URL changes, go to:
      github.com/RITIK-12/CricketShotClassification → Releases
    and copy the .pth download link, then update PRETRAINED_URL above.
    """
    dest_path = Path(dest)
    dest_path.parent.mkdir(parents=True, exist_ok=True)

    if dest_path.exists():
        print(f"[weights] Already downloaded: {dest}")
        return dest

    print(f"[weights] Downloading pretrained weights from RITIK-12 repo...")
    print(f"          → {PRETRAINED_URL}")
    try:
        urllib.request.urlretrieve(PRETRAINED_URL, dest, _download_progress)
        print(f"\n[weights] Saved to {dest}")
    except Exception as e:
        print(f"\n[weights] Auto-download failed: {e}")
        print("\nManual download steps:")
        print("  1. Go to: github.com/RITIK-12/CricketShotClassification")
        print("  2. Click Releases → download the .pth file")
        print(f"  3. Copy it to: {dest}")
    return dest


def _download_progress(block, block_size, total):
    downloaded = block * block_size
    if total > 0:
        pct = min(100, downloaded * 100 // total)
        bar = "█" * (pct // 5) + "░" * (20 - pct // 5)
        print(f"\r  [{bar}] {pct}%", end="", flush=True)


def load_pretrained(model: ShotClassifier, checkpoint: str = PRETRAINED_PATH) -> ShotClassifier:
    """
    Load RITIK-12 weights into our model.

    Uses strict=False so minor architecture differences (e.g. slightly
    different head size in the original repo) don't cause a hard crash —
    matching layers load, mismatching layers keep random init.
    """
    if not Path(checkpoint).exists():
        raise FileNotFoundError(
            f"Checkpoint not found at '{checkpoint}'.\n"
            "Run: python src/shot_classifier.py --download-weights"
        )

    state = torch.load(checkpoint, map_location="cpu")

    # Handle checkpoints saved as {"model_state_dict": ..., "epoch": ...}
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    elif isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]

    result = model.load_state_dict(state, strict=False)

    loaded   = len(state) - len(result.missing_keys)
    total    = len(list(model.state_dict()))
    print(f"[load] Loaded {loaded}/{total} layers from checkpoint")
    if result.missing_keys:
        print(f"       Missing (will use random init): {result.missing_keys[:4]}{'...' if len(result.missing_keys)>4 else ''}")
    if result.unexpected_keys:
        print(f"       Unexpected (skipped): {result.unexpected_keys[:4]}")

    return model




def download_cricshot10(dest_dir: str = DATA_ROOT):
    """
    CricShot10 is the smaller original dataset (~150 clips/class).
    Clone from: github.com/ascuet/CricShot10

    Manual steps (git clone is easiest):
      git clone https://github.com/ascuet/CricShot10.git data/CricShot10

    Expected structure after clone:
      data/CricShot10/
        train/ val/ test/
          cover_drive/  *.mp4
          straight_drive/ ...
    """
    dest = Path(dest_dir)
    if dest.exists() and any(dest.iterdir()):
        print(f"[data] Dataset already present at {dest_dir}")
        return

    print("[data] CricShot10 not found. Clone it with:")
    print(f"  git clone https://github.com/ascuet/CricShot10.git {dest_dir}")
    print("\nExpected structure:")
    print("  data/CricShot10/train/cover_drive/*.mp4")
    print("  data/CricShot10/train/straight_drive/*.mp4  ... (10 classes)")
    print("  data/CricShot10/val/   ... same")
    print("  data/CricShot10/test/  ... same")


class CricShotDataset(torch.utils.data.Dataset):
    """
    Loads video clips from the CricShot10 directory structure.
    Each item returns (clip_tensor, label_index).
    clip_tensor shape: (T, 3, 224, 224)
    """

    def __init__(self, root: str = DATA_ROOT, split: str = "train",
                 n_frames: int = NUM_FRAMES, augment: bool = True):
        self.split    = split
        self.n_frames = n_frames
        self.augment  = augment and (split == "train")
        self.samples: list[tuple[Path, int]] = []

        split_dir = Path(root) / split
        if not split_dir.exists():
            raise FileNotFoundError(
                f"Split directory not found: {split_dir}\n"
                "Run: python src/shot_classifier.py --download-data"
            )

        for cls_idx, cls_name in enumerate(SHOT_CLASSES):
            cls_dir = split_dir / cls_name
            if not cls_dir.exists():
                continue
            for ext in ("*.mp4", "*.avi", "*.mov"):
                for vid in cls_dir.glob(ext):
                    self.samples.append((vid, cls_idx))

        if not self.samples:
            raise RuntimeError(
                f"No video files found under {split_dir}.\n"
                "Check that class subdirectories exist and contain .mp4/.avi files."
            )
        print(f"[dataset] {split}: {len(self.samples)} clips across "
              f"{len(set(s[1] for s in self.samples))} classes")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        vid_path, label = self.samples[idx]
        frames = _sample_frames(str(vid_path), self.n_frames)

        # Augmentation: horizontal flip (handles left-handed batsmen)
        if self.augment and np.random.rand() > 0.5:
            frames = [np.fliplr(f) for f in frames]

        tensor = _preprocess(frames)   # (T, 3, 224, 224)
        return tensor, label




def finetune(
    data_root:      str   = DATA_ROOT,
    pretrained:     str   = PRETRAINED_PATH,
    save_path:      str   = FINETUNED_PATH,
    epochs:         int   = 5,
    batch_size:     int   = 8,
    lr:             float = 3e-4,
    unfreeze_epoch: int   = 3,    # after this epoch, unfreeze everything for full fine-tune
):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[finetune] Device: {device}")

   
    train_ds = CricShotDataset(data_root, "train", augment=True)
    val_ds   = CricShotDataset(data_root, "val",   augment=False)
    train_dl = torch.utils.data.DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=min(4, os.cpu_count() or 1), pin_memory=device.type == "cuda"
    )
    val_dl = torch.utils.data.DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=min(4, os.cpu_count() or 1)
    )

    model = ShotClassifier().to(device)
    model = load_pretrained(model, pretrained)
    model.freeze_backbone()   # only head trains initially

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr, weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    best_val_acc = 0.0

    for epoch in range(1, epochs + 1):

        # Phase switch: unfreeze all layers mid-training
        if epoch == unfreeze_epoch + 1:
            print(f"\n[finetune] Epoch {epoch}: unfreezing all layers (LR ÷ 10)")
            model.unfreeze_all()
            for pg in optimizer.param_groups:
                pg["lr"] = lr / 10
            # Re-add all params to optimizer
            optimizer = torch.optim.AdamW(
                model.parameters(), lr=lr / 10, weight_decay=1e-4
            )

       
        model.train()
        train_loss, correct, total = 0.0, 0, 0

        for clips, labels in train_dl:
            clips, labels = clips.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(clips)
            loss   = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()
            correct    += (logits.argmax(1) == labels).sum().item()
            total      += labels.size(0)

        train_acc = correct / total

     
        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for clips, labels in val_dl:
                clips, labels = clips.to(device), labels.to(device)
                val_correct += (model(clips).argmax(1) == labels).sum().item()
                val_total   += labels.size(0)

        val_acc = val_correct / val_total
        scheduler.step()

        phase = "HEAD" if epoch <= unfreeze_epoch else "FULL"
        print(f"Epoch {epoch:2d}/{epochs} [{phase}] | "
              f"loss={train_loss/len(train_dl):.4f} | "
              f"train={train_acc:.4f} | val={val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), save_path)
            print(f"  ✓ Best model saved  (val_acc={val_acc:.4f})")

    print(f"\n[finetune] Complete. Best val accuracy: {best_val_acc:.4f}")
    print(f"[finetune] Weights saved to: {save_path}")
    return best_val_acc


def train_from_scratch(
    data_root:  str   = DATA_ROOT,
    save_path:  str   = FINETUNED_PATH,
    epochs:     int   = 40,
    batch_size: int   = 8,
    lr:         float = 1e-4,
):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[train] Full training from scratch on {device}")

    train_ds = CricShotDataset(data_root, "train")
    val_ds   = CricShotDataset(data_root, "val")
    train_dl = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=4)
    val_dl   = torch.utils.data.DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=4)

    model     = ShotClassifier().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    best_val_acc = 0.0

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss, correct, total = 0.0, 0, 0
        for clips, labels in train_dl:
            clips, labels = clips.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(clips)
            loss   = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()
            correct    += (logits.argmax(1) == labels).sum().item()
            total      += labels.size(0)

        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for clips, labels in val_dl:
                clips, labels = clips.to(device), labels.to(device)
                val_correct += (model(clips).argmax(1) == labels).sum().item()
                val_total   += labels.size(0)

        val_acc = val_correct / val_total
        scheduler.step()
        print(f"Epoch {epoch:3d}/{epochs} | loss={train_loss/len(train_dl):.4f} | "
              f"train={correct/total:.4f} | val={val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), save_path)
            print(f"  ✓ Saved (val={val_acc:.4f})")

    print(f"\nDone. Best val accuracy: {best_val_acc:.4f}")




class ShotInference:
   

    def __init__(self, weights: str = FINETUNED_PATH, device: str | None = None):
        self.device = torch.device(
            device if device
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        if not Path(weights).exists():
            raise FileNotFoundError(
                f"No weights at '{weights}'.\n"
                "Run the fine-tune step first:\n"
                "  python src/shot_classifier.py --finetune"
            )
        self.model = ShotClassifier()
        state = torch.load(weights, map_location=self.device)
        self.model.load_state_dict(state)
        self.model.to(self.device)
        self.model.eval()
        print(f"[inference] Model loaded from {weights} on {self.device}")

    def classify_video(self, video_path: str) -> dict:
        """
        Uniformly sample NUM_FRAMES from the video, run the model.

        Returns:
          {
            "shot":       "cover_drive",
            "confidence": 0.87,
            "all_probs":  {"cover_drive": 0.87, "straight_drive": 0.05, ...}
          }
        """
        frames = _sample_frames(video_path, NUM_FRAMES)
        tensor = _preprocess(frames).unsqueeze(0).to(self.device)  # (1, T, C, H, W)

        with torch.no_grad():
            probs = F.softmax(self.model(tensor), dim=1)[0]   # (10,)

        idx  = probs.argmax().item()
        conf = probs[idx].item()

        return {
            "shot":       SHOT_CLASSES[idx],
            "confidence": round(conf, 4),
            "all_probs":  {
                SHOT_CLASSES[i]: round(probs[i].item(), 4)
                for i in range(NUM_CLASSES)
            },
        }

    def top_k(self, video_path: str, k: int = 3) -> list[dict]:
        """Return top-k predictions sorted by confidence."""
        result = self.classify_video(video_path)
        sorted_probs = sorted(result["all_probs"].items(), key=lambda x: -x[1])
        return [{"shot": s, "confidence": p} for s, p in sorted_probs[:k]]




def _sample_frames(video_path: str, n: int) -> list[np.ndarray]:
    """Uniformly sample n RGB frames from a video file."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_path}")

    total   = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    indices = np.linspace(0, total - 1, n, dtype=int)
    frames  = []

    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if ret:
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    cap.release()

    # Pad to exactly n frames if video was short
    while len(frames) < n:
        frames.append(
            frames[-1].copy() if frames
            else np.zeros((224, 224, 3), dtype=np.uint8)
        )
    return frames[:n]


def _preprocess(frames: list[np.ndarray]) -> torch.Tensor:
    """Transform a list of RGB frames to (T, 3, 224, 224)."""
    return torch.stack([_transform(f) for f in frames])


#

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Cricket shot classifier — pretrained weight fine-tuning approach"
    )
    parser.add_argument("--download-weights", action="store_true",
                        help="Download RITIK-12 pretrained .pth checkpoint")
    parser.add_argument("--download-data",    action="store_true",
                        help="Print instructions to download CricShot10 dataset")
    parser.add_argument("--finetune",         action="store_true",
                        help="Fine-tune head on CricShot10 (recommended, ~20 min)")
    parser.add_argument("--train-scratch",    action="store_true",
                        help="Full training from ImageNet init (~4–8 hrs)")
    parser.add_argument("--infer",            type=str, metavar="VIDEO",
                        help="Run inference on a video file")
    parser.add_argument("--top-k",            type=int, default=3,
                        help="Number of top predictions to show (default 3)")
    parser.add_argument("--data",             type=str, default=DATA_ROOT)
    parser.add_argument("--epochs",           type=int, default=5)
    parser.add_argument("--weights",          type=str, default=FINETUNED_PATH)
    args = parser.parse_args()

    if args.download_weights:
        download_pretrained_weights()

    elif args.download_data:
        download_cricshot10(args.data)

    elif args.finetune:
        if not Path(PRETRAINED_PATH).exists():
            print("[!] Pretrained weights not found — downloading first...")
            download_pretrained_weights()
        finetune(data_root=args.data, epochs=args.epochs, save_path=args.weights)

    elif args.train_scratch:
        train_from_scratch(data_root=args.data, epochs=args.epochs, save_path=args.weights)

    elif args.infer:
        inf    = ShotInference(weights=args.weights)
        result = inf.classify_video(args.infer)
        print(f"\nShot       : {result['shot'].replace('_', ' ').title()}")
        print(f"Confidence : {result['confidence']:.2%}")
        print(f"\nTop {args.top_k}:")
        for pred in inf.top_k(args.infer, k=args.top_k):
            bar = "█" * int(pred['confidence'] * 20)
            print(f"  {pred['shot']:<18}  {pred['confidence']:.2%}  {bar}")

    else:
        parser.print_help()
        print("\nQuick start:")
        print("  python src/shot_classifier.py --download-weights")
        print("  git clone https://github.com/ascuet/CricShot10.git data/CricShot10")
        print("  python src/shot_classifier.py --finetune")
        print("  python src/shot_classifier.py --infer your_clip.mp4")