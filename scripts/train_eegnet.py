"""Train and evaluate one subject/paradigm/fold/seed EEGNet task."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from privileged_emg.eegnet import EEGNet82, architecture_dict


def read_jsonl_row(path: Path, index: int) -> dict:
    with path.open() as handle:
        for row_index, line in enumerate(handle):
            if row_index == index:
                return json.loads(line)
    raise IndexError(f"{path}: no row {index}")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


class IndexedEEGDataset(Dataset):
    def __init__(self, eeg_path: Path, label_path: Path, indices: np.ndarray) -> None:
        self.eeg = np.load(eeg_path, mmap_mode="r")
        raw_labels = np.load(label_path, mmap_mode="r")
        self.indices = np.asarray(indices, dtype=np.int64)

        if self.eeg.ndim != 3 or tuple(self.eeg.shape[1:]) != (60, 256):
            raise ValueError(f"{eeg_path}: expected (N,60,256), got {self.eeg.shape}")
        if len(raw_labels) != len(self.eeg):
            raise ValueError(f"EEG/label length mismatch: {len(self.eeg)} != {len(raw_labels)}")
        if self.indices.ndim != 1 or len(self.indices) == 0:
            raise ValueError("Split indices must be a non-empty 1-D array")
        if self.indices.min() < 0 or self.indices.max() >= len(self.eeg):
            raise ValueError("Split contains an out-of-range trial index")

        labels = np.asarray(raw_labels, dtype=np.int64)
        if labels.min() == 1 and labels.max() == 10:
            labels = labels - 1
        if labels.min() < 0 or labels.max() > 9:
            raise ValueError(f"Expected labels 0..9 or 1..10, got {labels.min()}..{labels.max()}")
        self.labels = labels

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int):
        trial_index = int(self.indices[item])
        # Copy prevents a read-only NumPy memmap from being exposed to PyTorch.
        eeg = np.array(self.eeg[trial_index], dtype=np.float32, copy=True)
        return torch.from_numpy(eeg), int(self.labels[trial_index]), trial_index


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[dict, np.ndarray]:
    matrix = np.zeros((10, 10), dtype=np.int64)
    for true, pred in zip(y_true, y_pred):
        matrix[int(true), int(pred)] += 1
    support = matrix.sum(axis=1)
    predicted = matrix.sum(axis=0)
    true_positive = np.diag(matrix).astype(np.float64)
    recall = np.divide(true_positive, support, out=np.zeros(10), where=support > 0)
    precision = np.divide(true_positive, predicted, out=np.zeros(10), where=predicted > 0)
    f1 = np.divide(
        2 * precision * recall,
        precision + recall,
        out=np.zeros(10),
        where=(precision + recall) > 0,
    )
    present = support > 0
    return {
        "n": int(len(y_true)),
        "accuracy": float(100.0 * np.mean(y_true == y_pred)),
        "balanced_accuracy": float(100.0 * recall[present].mean()),
        "macro_f1": float(100.0 * f1[present].mean()),
        "n_present_classes": int(present.sum()),
    }, matrix


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device):
    model.eval()
    labels_all, predictions_all, probabilities_all, indices_all = [], [], [], []
    for eeg, labels, indices in loader:
        logits = model(eeg.to(device, non_blocking=True))
        probabilities = torch.softmax(logits, dim=1)
        labels_all.append(labels.numpy())
        predictions_all.append(probabilities.argmax(dim=1).cpu().numpy())
        probabilities_all.append(probabilities.cpu().numpy())
        indices_all.append(indices.numpy())
    y_true = np.concatenate(labels_all)
    y_pred = np.concatenate(predictions_all)
    probs = np.concatenate(probabilities_all)
    trial_indices = np.concatenate(indices_all)
    metrics, confusion = classification_metrics(y_true, y_pred)
    return metrics, confusion, y_true, y_pred, probs, trial_indices


def make_loader(dataset: Dataset, batch_size: int, shuffle: bool, seed: int, workers: int):
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator if shuffle else None,
        num_workers=workers,
        pin_memory=True,
        drop_last=shuffle,
        persistent_workers=workers > 0,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--index", type=int, required=True)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    task = read_jsonl_row(args.manifest, args.index)
    out_dir = Path(task["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    result_path = out_dir / "result.json"
    if result_path.is_file():
        try:
            existing = json.loads(result_path.read_text())
            if existing.get("status") == "complete" and (out_dir / "predictions.npz").is_file():
                print(f"SKIP complete task: {out_dir}", flush=True)
                return
        except (OSError, json.JSONDecodeError):
            pass

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the formal EEGNet run")

    seed = int(task["seed"])
    fold = int(task["fold"])
    subject = task["subject"]
    paradigm = task["paradigm"]
    set_seed(seed)
    device = torch.device("cuda:0")

    eeg_path = Path(task["eeg_path"])
    label_path = Path(task["label_path"])
    split_path = Path(task["split_file"])
    for required in (eeg_path, label_path, split_path):
        if not required.is_file():
            raise FileNotFoundError(required)

    split = np.load(split_path)
    train_idx = np.asarray(split["train_idx"], dtype=np.int64)
    val_idx = np.asarray(split["val_idx"], dtype=np.int64)
    if set(train_idx) & set(val_idx):
        raise RuntimeError("Train/validation split leakage detected")

    train_set = IndexedEEGDataset(eeg_path, label_path, train_idx)
    val_set = IndexedEEGDataset(eeg_path, label_path, val_idx)
    loader_seed = seed + 1000 * fold + int(subject[1:])
    train_loader = make_loader(train_set, int(task["batch_size"]), True, loader_seed, args.workers)
    val_loader = make_loader(val_set, int(task["eval_batch_size"]), False, loader_seed, args.workers)

    model = EEGNet82().to(device)
    architecture = architecture_dict(model)
    criterion = nn.CrossEntropyLoss(label_smoothing=float(task["label_smoothing"]))
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(task["learning_rate"]),
        weight_decay=float(task["weight_decay"]),
    )
    epochs = int(task["epochs"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=float(task["eta_min"])
    )

    metadata = {
        **task,
        "manifest": str(args.manifest),
        "manifest_sha256": sha256(args.manifest),
        "split_sha256": sha256(split_path),
        "architecture": architecture,
        "torch_version": torch.__version__,
        "cuda_device": torch.cuda.get_device_name(0),
        "selection_rule": "maximum validation balanced accuracy; test evaluated once after selection",
    }
    (out_dir / "meta.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False))

    best_val_ba = -1.0
    best_epoch = 0
    checkpoint_path = out_dir / "best_eegnet.pth"
    curve = []
    started = time.time()

    for epoch in range(epochs):
        model.train()
        loss_total = 0.0
        n_seen = 0
        n_correct = 0
        for eeg, labels, _ in train_loader:
            eeg = eeg.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model(eeg)
            loss = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(task["gradient_clip"]))
            optimizer.step()
            model.apply_max_norm()
            loss_total += float(loss.item()) * len(labels)
            n_seen += len(labels)
            n_correct += int((logits.argmax(dim=1) == labels).sum().item())
        scheduler.step()

        val_metrics, _, _, _, _, _ = evaluate(model, val_loader, device)
        curve.append({
            "epoch": epoch + 1,
            "train_loss": loss_total / max(n_seen, 1),
            "train_accuracy": 100.0 * n_correct / max(n_seen, 1),
            "learning_rate": optimizer.param_groups[0]["lr"],
            **{f"val_{key}": value for key, value in val_metrics.items()},
        })
        if val_metrics["balanced_accuracy"] > best_val_ba:
            best_val_ba = val_metrics["balanced_accuracy"]
            best_epoch = epoch + 1
            torch.save(model.state_dict(), checkpoint_path)
        if epoch == 0 or (epoch + 1) % 10 == 0:
            print(
                f"epoch={epoch+1:03d}/{epochs} "
                f"loss={curve[-1]['train_loss']:.4f} "
                f"val_BA={val_metrics['balanced_accuracy']:.4f} "
                f"best={best_val_ba:.4f}@{best_epoch}",
                flush=True,
            )

    with (out_dir / "curve.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(curve[0]))
        writer.writeheader()
        writer.writerows(curve)

    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state, strict=True)
    val_metrics, _, _, _, _, _ = evaluate(model, val_loader, device)

    # Final-test indices and loader are constructed only after validation
    # selection and strict reload of the selected checkpoint.
    test_idx = np.asarray(split["test_idx"], dtype=np.int64)
    if set(train_idx) & set(test_idx) or set(val_idx) & set(test_idx):
        raise RuntimeError("Final-test split leakage detected")
    test_set = IndexedEEGDataset(eeg_path, label_path, test_idx)
    test_loader = make_loader(
        test_set, int(task["eval_batch_size"]), False, loader_seed, args.workers
    )
    test_metrics, confusion, y_true, y_pred, probs, trial_indices = evaluate(
        model, test_loader, device
    )
    np.savez_compressed(
        out_dir / "predictions.npz",
        y_true=y_true,
        y_pred=y_pred,
        probabilities=probs,
        trial_indices=trial_indices,
        confusion_matrix=confusion,
    )
    result = {
        **task,
        "status": "complete",
        "n_train": len(train_set),
        "n_val": len(val_set),
        "n_test": len(test_set),
        "best_epoch": best_epoch,
        "best_val_balanced_accuracy": best_val_ba,
        **{f"selected_val_{key}": value for key, value in val_metrics.items()},
        **{f"test_{key}": value for key, value in test_metrics.items()},
        "architecture": architecture,
        "checkpoint": str(checkpoint_path),
        "predictions": str(out_dir / "predictions.npz"),
        "elapsed_seconds": time.time() - started,
    }
    result_path.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
