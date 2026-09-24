from pathlib import Path
import sys
import argparse
import csv
import hashlib
import json
import os

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader



from .common import (
    read_jsonl,
    save_json,
    set_seed,
    require_data_root,
    load_unseen_subjects,
    subset_dict,
    classification_metrics,
)

from .dataset import BCIDataset
from .model_stage1 import RobustEMGPretrainer
from .model_stage2 import EEGMAEAlignerPretrainer, NoDistillPretrainer
from .model_stage3 import EEGDownstreamClassifier, EEGZeroShotClassifier
from .label_simplex import (
    EEGIndexedDataset,
    FinalTestBoundary,
    LabelSimplexPretrainer,
    install_emg_access_guard,
    load_eeg_label_train_validation,
    load_partition_indices,
    prototype_npy_bytes,
    regular_simplex_prototypes,
    stage3_data_paths,
)


SUBJECTS = [f"S{i:02d}" for i in range(1, 31)]
PARADIGMS = ["overt", "silent_1001"]

SPLIT_ROOT = None

KEYS = [
    "eeg",
    "emg",
    "audio",
    "labels",
    "is_overt",
]


def load_state(model, path):
    state = torch.load(
        path,
        map_location="cpu"
    )

    state = {
        k.replace("module.", "", 1): v
        for k, v in state.items()
    }

    model.load_state_dict(
        state,
        strict=True
    )


def concat_dicts(parts):
    if not parts:
        raise ValueError("No data parts")

    return {
        k: np.concatenate(
            [p[k] for p in parts],
            axis=0
        )
        for k in KEYS
    }


def configure(split_root):
    global SPLIT_ROOT
    SPLIT_ROOT = Path(split_root).expanduser().resolve()


def load_split(subject, paradigm, fold):
    if SPLIT_ROOT is None: raise RuntimeError("Split root is not configured")
    path = (
        SPLIT_ROOT /
        f"{subject}_{paradigm}_fold{fold}.npz"
    )

    z = np.load(path)

    return (
        z["train_idx"],
        z["val_idx"],
        z["test_idx"],
    )


def _derange_emg_within_class(
    data,
    rng,
    subject,
    paradigm,
):
    """Replace every EMG trial with a different same-class trial.

    This function is called on one subject/paradigm training partition at a
    time.  It therefore preserves subject, paradigm and lexical class while
    breaking exact EEG--EMG trial identity.  The mapping is a one-to-one
    derangement within every class: no EMG sample is duplicated or omitted.
    """
    labels = np.asarray(data["labels"])
    donor_idx = np.arange(len(labels), dtype=np.int64)
    class_sizes = []

    for label in np.unique(labels):
        members = np.flatnonzero(labels == label)
        n_members = len(members)
        class_sizes.append(n_members)

        if n_members < 2:
            raise ValueError(
                "Cannot construct a no-fixed-point within-class "
                f"derangement for {subject}/{paradigm}/class={int(label)}: "
                f"only {n_members} training trial(s)"
            )

        order = members.copy()
        rng.shuffle(order)
        shift = int(rng.integers(1, n_members))
        donor_idx[order] = np.roll(order, shift)

    fixed_points = int(np.sum(donor_idx == np.arange(len(labels))))
    label_mismatches = int(np.sum(labels != labels[donor_idx]))

    if fixed_points != 0 or label_mismatches != 0:
        raise AssertionError(
            f"Invalid derangement for {subject}/{paradigm}: "
            f"fixed_points={fixed_points}, "
            f"label_mismatches={label_mismatches}"
        )

    out = dict(data)
    out["emg"] = np.ascontiguousarray(
        data["emg"][donor_idx]
    )

    audit = {
        "subject": subject,
        "paradigm": paradigm,
        "n_trials": int(len(labels)),
        "n_classes": int(len(class_sizes)),
        "min_class_size": int(min(class_sizes)),
        "max_class_size": int(max(class_sizes)),
        "fixed_points": fixed_points,
        "label_mismatches": label_mismatches,
        "donor_idx": donor_idx,
    }

    return out, audit


def load_global_train_val(
    fold,
    pairing_variant="matched",
    pairing_seed=None,
    subjects=SUBJECTS,
):
    train_parts = []
    val_parts = []

    supported_pairings = {
        "matched",
        "within_subject_class_deranged",
    }

    if pairing_variant not in supported_pairings:
        raise ValueError(
            f"Unknown pairing_variant={pairing_variant!r}; "
            f"expected one of {sorted(supported_pairings)}"
        )

    if pairing_variant == "within_subject_class_deranged":
        if pairing_seed is None:
            raise ValueError(
                "pairing_seed is required for the derangement control"
            )
        pairing_rng = np.random.default_rng(
            int(pairing_seed)
        )
    else:
        pairing_rng = None

    pairing_rows = []
    mapping_hasher = hashlib.sha256()

    subjects = tuple(subjects)
    if not subjects or len(set(subjects)) != len(subjects):
        raise AssertionError("subjects must be nonempty and unique")
    if not set(subjects).issubset(SUBJECTS):
        raise AssertionError("unknown subject in upstream request")
    for subject in subjects:
        for paradigm in PARADIGMS:

            data = load_unseen_subjects(
                [subject],
                [paradigm]
            )

            tr, va, _ = load_split(
                subject,
                paradigm,
                fold,
            )

            train_part = subset_dict(data, tr)

            if pairing_rng is not None:
                train_part, pairing_row = (
                    _derange_emg_within_class(
                        train_part,
                        pairing_rng,
                        subject,
                        paradigm,
                    )
                )

                donor_idx = pairing_row.pop(
                    "donor_idx"
                )
                mapping_hasher.update(
                    subject.encode("utf-8")
                )
                mapping_hasher.update(
                    paradigm.encode("utf-8")
                )
                mapping_hasher.update(
                    donor_idx.astype(
                        "<i8",
                        copy=False,
                    ).tobytes()
                )
                pairing_rows.append(pairing_row)

            train_parts.append(train_part)

            val_parts.append(
                subset_dict(data, va)
            )

    train_data = concat_dicts(train_parts)
    val_data = concat_dicts(val_parts)

    if pairing_rng is None:
        pairing_audit = {
            "pairing_variant": "matched",
            "applied": False,
            "scope": "original trial-matched EEG--EMG pairs",
            "n_train": int(len(train_data["labels"])),
        }
    else:
        pairing_audit = {
            "pairing_variant": pairing_variant,
            "applied": True,
            "pairing_seed": int(pairing_seed),
            "scope": (
                "training partition only; one-to-one EMG derangement "
                "within each subject, paradigm, and lexical class"
            ),
            "n_train": int(len(train_data["labels"])),
            "n_subject_paradigm_strata": int(len(pairing_rows)),
            "n_class_strata": int(
                sum(x["n_classes"] for x in pairing_rows)
            ),
            "min_class_size": int(
                min(x["min_class_size"] for x in pairing_rows)
            ),
            "max_class_size": int(
                max(x["max_class_size"] for x in pairing_rows)
            ),
            "fixed_points": int(
                sum(x["fixed_points"] for x in pairing_rows)
            ),
            "label_mismatches": int(
                sum(x["label_mismatches"] for x in pairing_rows)
            ),
            "subject_mismatches": 0,
            "paradigm_mismatches": 0,
            "mapping_sha256": mapping_hasher.hexdigest(),
        }

        if (
            pairing_audit["fixed_points"] != 0
            or pairing_audit["label_mismatches"] != 0
        ):
            raise AssertionError(pairing_audit)

    return train_data, val_data, pairing_audit


def load_subject_train_val(subject, paradigm, fold):
    data = load_unseen_subjects(
        [subject],
        [paradigm]
    )

    tr, va, _ = load_split(
        subject,
        paradigm,
        fold,
    )

    return (
        subset_dict(data, tr),
        subset_dict(data, va),
    )


def load_subject_test(subject, paradigm, fold):
    """Construct final-test data only after checkpoint selection and reload."""
    data = load_unseen_subjects([subject], [paradigm])
    _, _, test_idx = load_split(subject, paradigm, fold)
    return subset_dict(data, test_idx)


def load_subject_split(subject, paradigm, fold):
    """Compatibility helper for secondary analyses; primary Stage 3 is stricter."""
    train, val = load_subject_train_val(subject, paradigm, fold)
    return train, val, load_subject_test(subject, paradigm, fold)


def load_task_role_indices(task, role):
    if role not in {"train", "validation", "final_test"}:
        raise AssertionError(f"unknown split role: {role}")
    if "split_role_files" not in task:
        return load_partition_indices(
            SPLIT_ROOT, task["subject"], task["paradigm"], int(task["fold"]), role
        )
    path = Path(task["split_role_files"][role])
    if not path.is_absolute():
        path = Path(task["_manifest_dir"]) / path
    payload = path.read_bytes()
    if hashlib.sha256(payload).hexdigest() != task["split_role_sha256"][role]:
        raise AssertionError(f"{role} split-role hash mismatch")
    value = json.loads(payload)
    if set(value) != {"indices", "role"} or value["role"] != role:
        raise AssertionError(f"malformed {role} split-role file")
    indices = np.asarray(value["indices"], dtype=np.int64)
    if indices.ndim != 1 or not len(indices) or len(np.unique(indices)) != len(indices):
        raise AssertionError(f"malformed {role} split-role indices")
    return indices


@torch.no_grad()
def eval_stage1(model, loader, device):
    model.eval()

    old = model.emg_mask_ratio
    model.emg_mask_ratio = 0.0

    ys, ps, qs = [], [], []

    for _, emg, audio, y, overt in loader:
        z = model(
            emg.to(device),
            audio.to(device),
            y.to(device),
            overt.to(device),
        )

        p = torch.softmax(
            z["logits"],
            dim=1
        )

        ys.append(y.numpy())
        ps.append(
            p.argmax(1).cpu().numpy()
        )
        qs.append(
            p.cpu().numpy()
        )

    model.emg_mask_ratio = old

    return classification_metrics(
        np.concatenate(ys),
        np.concatenate(ps),
        np.concatenate(qs),
    )[0]


@torch.no_grad()
def eval_stage2(model, loader, device):
    model.eval()

    old = model.eeg_mask_ratio
    model.eeg_mask_ratio = 0.0

    ys, ps, qs = [], [], []

    for eeg, emg, _, y, _ in loader:
        z = model(
            eeg.to(device),
            emg.to(device),
            y.to(device),
            use_distill=False,
        )

        p = torch.softmax(
            z["logits"],
            dim=1
        )

        ys.append(y.numpy())
        ps.append(
            p.argmax(1).cpu().numpy()
        )
        qs.append(
            p.cpu().numpy()
        )

    model.eeg_mask_ratio = old

    return classification_metrics(
        np.concatenate(ys),
        np.concatenate(ps),
        np.concatenate(qs),
    )[0]


@torch.no_grad()
def eval_label_simplex(model, loader, device):
    model.eval()
    old = model.eeg_mask_ratio
    model.eeg_mask_ratio = 0.0
    ys, ps, qs = [], [], []
    for eeg, y in loader:
        z = model(eeg.to(device), y.to(device), use_label_loss=False)
        p = torch.softmax(z["logits"], dim=1)
        ys.append(y.numpy())
        ps.append(p.argmax(1).cpu().numpy())
        qs.append(p.cpu().numpy())
    model.eeg_mask_ratio = old
    return classification_metrics(
        np.concatenate(ys), np.concatenate(ps), np.concatenate(qs)
    )[0]


@torch.no_grad()
def eval_no_distill(model, loader, device):
    model.eval()
    old = model.eeg_mask_ratio
    model.eeg_mask_ratio = 0.0
    ys, ps, qs = [], [], []
    for eeg, y in loader:
        z = model(eeg.to(device), y.to(device))
        p = torch.softmax(z["logits"], dim=1)
        ys.append(y.numpy())
        ps.append(p.argmax(1).cpu().numpy())
        qs.append(p.cpu().numpy())
    model.eeg_mask_ratio = old
    return classification_metrics(
        np.concatenate(ys), np.concatenate(ps), np.concatenate(qs)
    )[0]


@torch.no_grad()
def eval_stage3(model, loader, device):
    model.eval()

    ys, ps, qs = [], [], []

    for batch in loader:
        eeg, y = (batch[0], batch[1]) if len(batch) == 2 else (batch[0], batch[3])
        out = model(
            eeg.to(device)
        )

        p = torch.softmax(
            out["logits"],
            dim=1
        )

        ys.append(y.numpy())
        ps.append(
            p.argmax(1).cpu().numpy()
        )
        qs.append(
            p.cpu().numpy()
        )

    return classification_metrics(
        np.concatenate(ys),
        np.concatenate(ps),
        np.concatenate(qs),
    )[0]


@torch.no_grad()
def eval_stage3_pair(adapted, zero_shot, loader, device):
    adapted.eval()
    zero_shot.eval()
    ys, adapted_ps, adapted_qs, zero_ps, zero_qs = [], [], [], [], []
    for batch in loader:
        eeg, y = (batch[0], batch[1]) if len(batch) == 2 else (batch[0], batch[3])
        eeg = eeg.to(device)
        adapted_prob = torch.softmax(adapted(eeg)["logits"], dim=1)
        zero_prob = torch.softmax(zero_shot(eeg)["logits"], dim=1)
        ys.append(y.numpy())
        adapted_ps.append(adapted_prob.argmax(1).cpu().numpy())
        adapted_qs.append(adapted_prob.cpu().numpy())
        zero_ps.append(zero_prob.argmax(1).cpu().numpy())
        zero_qs.append(zero_prob.cpu().numpy())
    truth = np.concatenate(ys)
    adapted_metrics = classification_metrics(
        truth, np.concatenate(adapted_ps), np.concatenate(adapted_qs)
    )[0]
    zero_metrics = classification_metrics(
        truth, np.concatenate(zero_ps), np.concatenate(zero_qs)
    )[0]
    return adapted_metrics, zero_metrics


def stage1(t):
    out = Path(t["out_dir"])
    out.mkdir(
        parents=True,
        exist_ok=True
    )

    save_json(
        out / "meta.json",
        t
    )

    if (out / "result.json").exists():
        return

    seed = int(t["seed"])
    fold = int(t["fold"])

    set_seed(seed)

    device = torch.device("cuda")

    trd, vad, _ = load_global_train_val(
        fold, subjects=t.get("upstream_subjects", SUBJECTS)
    )

    g = torch.Generator().manual_seed(
        seed
    )

    tr = DataLoader(
        BCIDataset(trd),
        batch_size=64,
        shuffle=True,
        generator=g,
        num_workers=6,
        drop_last=True,
    )

    va = DataLoader(
        BCIDataset(vad),
        batch_size=128,
        shuffle=False,
        num_workers=4,
    )

    model = RobustEMGPretrainer(
        256,
        10,
        0.75,
    ).to(device)

    opt = optim.AdamW(
        model.parameters(),
        lr=5e-4,
        weight_decay=0.05,
    )

    epochs = int(t["epochs"])

    sch = optim.lr_scheduler.CosineAnnealingLR(
        opt,
        T_max=epochs,
        eta_min=1e-6,
    )

    best = -1.0
    best_ep = 0

    bp = out / "best_stage1.pth"
    curve = []

    for ep in range(epochs):

        model.train()

        sums = {
            "mae": 0.0,
            "ce": 0.0,
            "align": 0.0,
            "total": 0.0,
        }

        for _, emg, audio, y, overt in tr:

            emg = emg.to(device)
            audio = audio.to(device)
            y = y.to(device)
            overt = overt.to(device)

            opt.zero_grad(
                set_to_none=True
            )

            z = model(
                emg,
                audio,
                y,
                overt,
            )

            loss = (
                z["loss_mae"]
                + 0.1 * z["loss_ce"]
                + z["loss_align"]
            )

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                1.0,
            )

            opt.step()

            sums["mae"] += z["loss_mae"].item()
            sums["ce"] += z["loss_ce"].item()
            sums["align"] += z["loss_align"].item()
            sums["total"] += loss.item()

        sch.step()

        m = eval_stage1(
            model,
            va,
            device,
        )

        row = {
            "epoch": ep + 1,
            **{
                k: v / max(len(tr), 1)
                for k, v in sums.items()
            },
            **{
                f"val_{k}": v
                for k, v in m.items()
            },
        }

        curve.append(row)

        if m["balanced_accuracy"] > best:
            best = m["balanced_accuracy"]
            best_ep = ep + 1
            torch.save(
                model.state_dict(),
                bp
            )

        if (
            ep == 0
            or (ep + 1) % 5 == 0
        ):
            print(
                row,
                "best",
                best,
                flush=True,
            )

    with (
        out / "curve.csv"
    ).open("w", newline="") as f:

        w = csv.DictWriter(
            f,
            fieldnames=curve[0].keys()
        )

        w.writeheader()
        w.writerows(curve)

    save_json(
        out / "result.json",
        {
            **t,
            "n_train": len(trd["labels"]),
            "n_val": len(vad["labels"]),
            "best_val_balanced_accuracy": best,
            "best_epoch": best_ep,
            "best_checkpoint": str(bp),
        },
    )


def make_matched_shuffle(emg, generator):
    """
    Random cyclic derangement:
    - same batch size
    - no fixed point when B>1
    - does not consume default CUDA RNG
    """
    B = len(emg)

    if B <= 1:
        return emg

    shift = int(
        torch.randint(
            low=1,
            high=B,
            size=(1,),
            generator=generator,
            device=emg.device,
        ).item()
    )

    idx = torch.arange(
        B,
        device=emg.device
    )

    idx = torch.roll(
        idx,
        shifts=shift
    )

    return emg[idx]


def stage2_label_simplex(t):
    """Run the frozen EEG/label-only LabelSimplex Stage-2 protocol."""
    out = Path(t["out_dir"])
    out.mkdir(parents=True, exist_ok=True)
    save_json(out / "meta.json", t)
    if (out / "result.json").exists():
        return
    if SPLIT_ROOT is None:
        raise RuntimeError("Split root is not configured")
    if t.get("stage1_ckpt") is not None:
        raise AssertionError("LabelSimplex must not receive a Stage-1 checkpoint")

    install_emg_access_guard()
    seed, fold = int(t["seed"]), int(t["fold"])
    set_seed(seed)
    device = torch.device("cuda")
    trd, vad, data_audit = load_eeg_label_train_validation(
        require_data_root(), SPLIT_ROOT, fold
    )
    save_json(out / "eeg_only_data_audit.json", data_audit)

    generator = torch.Generator().manual_seed(seed)
    tr = DataLoader(
        trd, batch_size=8, shuffle=True, generator=generator,
        num_workers=6, drop_last=True,
    )
    va = DataLoader(vad, batch_size=64, shuffle=False, num_workers=4)
    prototypes = torch.from_numpy(regular_simplex_prototypes())
    prototype_sha256 = hashlib.sha256(prototype_npy_bytes()).hexdigest()
    model = LabelSimplexPretrainer(prototypes, 256, 10, 0.75).to(device)
    opt = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.05)
    epochs = int(t["epochs"])
    sch = optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=epochs, eta_min=1e-6
    )
    best, best_ep = -1.0, 0
    bp = out / "best_stage2.pth"
    curve = []
    for ep in range(epochs):
        model.train()
        sums = {"mae": 0.0, "label": 0.0, "ce": 0.0, "total": 0.0}
        post = ep >= 30
        for eeg, y in tr:
            eeg, y = eeg.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            z = model(eeg, y, use_label_loss=post)
            loss = z["loss_mae"] if not post else (
                z["loss_mae"] + z["loss_label"] + 0.1 * z["loss_ce"]
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            for name, key in (
                ("mae", "loss_mae"), ("label", "loss_label"),
                ("ce", "loss_ce")
            ):
                sums[name] += z[key].item()
            sums["total"] += loss.item()
        sch.step()
        metrics = eval_label_simplex(model, va, device)
        row = {
            "epoch": ep + 1,
            **{key: value / max(len(tr), 1) for key, value in sums.items()},
            **{f"val_{key}": value for key, value in metrics.items()},
        }
        curve.append(row)
        if metrics["balanced_accuracy"] > best:
            best, best_ep = metrics["balanced_accuracy"], ep + 1
            torch.save(model.state_dict(), bp)
        if ep == 0 or (ep + 1) % 5 == 0:
            print(row, "best", best, flush=True)

    with (out / "curve.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=curve[0].keys())
        writer.writeheader()
        writer.writerows(curve)
    save_json(out / "result.json", {
        **t,
        "split_family": "outer_seed_20260811",
        "prototype_sha256": prototype_sha256,
        "n_train": len(trd),
        "n_val": len(vad),
        "best_val_balanced_accuracy": best,
        "best_epoch": best_ep,
        "best_checkpoint": str(bp),
        "emg_trial_files_opened": 0,
        "emg_checkpoints_opened": 0,
        "checkpoint_selection": "validation_balanced_accuracy",
    })


def stage2(t):
    if t["variant"] == "label_simplex":
        return stage2_label_simplex(t)

    out = Path(t["out_dir"])
    out.mkdir(
        parents=True,
        exist_ok=True
    )

    save_json(
        out / "meta.json",
        t
    )

    if (out / "result.json").exists():
        return

    variant_aliases = {
        "unrestricted_shuffle": "shuffled_emg_matched",
        "within_class_shuffle": "within_subject_class_deranged",
    }
    no_distill = t["variant"] == "no_distill"
    if not no_distill and not Path(
        t["stage1_ckpt"]
    ).is_file():
        raise FileNotFoundError(
            t["stage1_ckpt"]
        )

    seed = int(t["seed"])
    fold = int(t["fold"])
    variant = variant_aliases.get(t["variant"], t["variant"])

    set_seed(seed)

    device = torch.device("cuda")

    if variant == "within_subject_class_deranged":
        pairing_variant = variant
        pairing_seed = (
            seed
            + 1000003 * fold
            + 0x5EEDC0DE
        )
    else:
        pairing_variant = "matched"
        pairing_seed = None

    upstream_subjects = t.get("upstream_subjects", SUBJECTS)
    if no_distill:
        trd, vad, eeg_only_audit = load_eeg_label_train_validation(
            require_data_root(), SPLIT_ROOT, fold, upstream_subjects
        )
        pairing_audit = {
            "pairing_variant": "no_distill",
            "applied": False,
            "emg_access": "prohibited",
            "eeg_only_data_audit": eeg_only_audit,
        }
    else:
        trd, vad, pairing_audit = load_global_train_val(
            fold,
            pairing_variant=pairing_variant,
            pairing_seed=pairing_seed,
            subjects=upstream_subjects,
        )

    save_json(
        out / "emg_pairing_audit.json",
        pairing_audit,
    )

    g = torch.Generator().manual_seed(
        seed
    )

    tr = DataLoader(
        trd if no_distill else BCIDataset(trd),
        batch_size=8,
        shuffle=True,
        generator=g,
        num_workers=6,
        drop_last=True,
    )

    va = DataLoader(
        vad if no_distill else BCIDataset(vad),
        batch_size=64,
        shuffle=False,
        num_workers=4,
    )

    weak_weight = float(
        t["weak_weight"]
    )

    model = (
        NoDistillPretrainer(256, 10, 0.75)
        if no_distill else
        EEGMAEAlignerPretrainer(
            t["stage1_ckpt"], 256, 10, 0.75, weak_weight=weak_weight
        )
    ).to(device)

    params = [
        p
        for p in model.parameters()
        if p.requires_grad
    ]

    opt = optim.AdamW(
        params,
        lr=3e-4,
        weight_decay=0.05,
    )

    epochs = int(t["epochs"])

    sch = optim.lr_scheduler.CosineAnnealingLR(
        opt,
        T_max=epochs,
        eta_min=1e-6,
    )

    # Dedicated RNG:
    # shuffle control does not perturb default
    # CUDA RNG used by masking/dropout.
    shuffle_gen = torch.Generator(
        device="cuda"
    )

    shuffle_gen.manual_seed(
        seed + 0x5EED
    )

    best = -1.0
    best_ep = 0

    bp = out / "best_stage2.pth"
    curve = []

    for ep in range(epochs):

        model.train()
        if not no_distill:
            model.emg_encoder.eval()

        sums = {
            "mae": 0.0,
            "ce": 0.0,
            "distill": 0.0,
            "total": 0.0,
        }

        for batch in tr:

            if no_distill:
                eeg, y = batch
                emg = None
            else:
                eeg, emg, _, y, _ = batch

            eeg = eeg.to(device)
            emg = emg.to(device) if emg is not None else None
            y = y.to(device)

            post = ep >= 30

            use_distill = (
                post
                and variant != "no_distill"
            )

            if (
                variant
                == "shuffled_emg_matched"
                and use_distill
            ):
                emg_in = make_matched_shuffle(
                    emg,
                    shuffle_gen,
                )
            else:
                emg_in = emg

            opt.zero_grad(
                set_to_none=True
            )

            z = (
                model(eeg, y)
                if no_distill else
                model(eeg, emg_in, y, use_distill=use_distill)
            )

            if not post:
                loss = z["loss_mae"]

            elif variant == "no_distill":
                loss = (
                    z["loss_mae"]
                    + 0.1 * z["loss_ce"]
                )

            else:
                loss = (
                    z["loss_mae"]
                    + z["loss_distill"]
                    + 0.1 * z["loss_ce"]
                )

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                params,
                1.0,
            )

            opt.step()

            sums["mae"] += z["loss_mae"].item()
            sums["ce"] += z["loss_ce"].item()
            sums["distill"] += 0.0 if no_distill else z["loss_distill"].item()
            sums["total"] += loss.item()

        sch.step()

        m = eval_no_distill(model, va, device) if no_distill else eval_stage2(model, va, device)

        row = {
            "epoch": ep + 1,
            **{
                k: v / max(len(tr), 1)
                for k, v in sums.items()
            },
            **{
                f"val_{k}": v
                for k, v in m.items()
            },
        }

        curve.append(row)

        if m["balanced_accuracy"] > best:
            best = m["balanced_accuracy"]
            best_ep = ep + 1
            torch.save(
                model.state_dict(),
                bp
            )

        if (
            ep == 0
            or (ep + 1) % 5 == 0
        ):
            print(
                row,
                "best",
                best,
                flush=True,
            )

    with (
        out / "curve.csv"
    ).open("w", newline="") as f:

        w = csv.DictWriter(
            f,
            fieldnames=curve[0].keys()
        )

        w.writeheader()
        w.writerows(curve)

    save_json(
        out / "result.json",
        {
            **t,
            "n_train": len(trd) if no_distill else len(trd["labels"]),
            "n_val": len(vad) if no_distill else len(vad["labels"]),
            "emg_pairing_audit": pairing_audit,
            "best_val_balanced_accuracy": best,
            "best_epoch": best_ep,
            "best_checkpoint": str(bp),
        },
    )


def stage3(t):
    out = Path(t["out_dir"])
    out.mkdir(
        parents=True,
        exist_ok=True
    )

    save_json(
        out / "meta.json",
        t
    )

    if (out / "result.json").exists():
        return

    ckpt = Path(
        t["stage2_ckpt"]
    )

    if not ckpt.is_file():
        raise FileNotFoundError(
            ckpt
        )

    seed = int(t["seed"])
    fold = int(t["fold"])

    # Same outer split for all variants.
    # Training randomness can vary by seed.
    set_seed(seed)

    label_simplex = t.get("variant") == "label_simplex"
    eeg_only = label_simplex or t.get("stage3_modality") == "EEG_only"
    if eeg_only:
        if SPLIT_ROOT is None:
            raise RuntimeError("Split root is not configured")
        if label_simplex:
            install_emg_access_guard()
        eeg_path, label_path = stage3_data_paths(
            require_data_root(), t["subject"], t["paradigm"]
        )
        train_idx = load_task_role_indices(t, "train")
        validation_idx = load_task_role_indices(t, "validation")
        if np.intersect1d(train_idx, validation_idx).size:
            raise AssertionError("Train/validation overlap")
        trd = EEGIndexedDataset(eeg_path, label_path, train_idx, "train")
        vad = EEGIndexedDataset(eeg_path, label_path, validation_idx, "validation")
    else:
        trd, vad = load_subject_train_val(
            t["subject"], t["paradigm"], fold,
        )

    device = torch.device("cuda")

    g = torch.Generator().manual_seed(
        seed
        + 1000 * fold
        + int(t["subject"][1:])
    )

    tr = DataLoader(
        trd if eeg_only else BCIDataset(trd),
        batch_size=8,
        shuffle=True,
        generator=g,
        num_workers=4,
        drop_last=True,
    )

    va = DataLoader(
        vad if eeg_only else BCIDataset(vad),
        batch_size=64,
        shuffle=False,
    )

    model = EEGDownstreamClassifier(
        str(ckpt),
        10,
    ).to(device)

    opt = optim.AdamW(
        [
            {
                "params": model.eeg_encoder.parameters(),
                "lr": float(t.get("encoder_lr", 3e-6)),
            },
            {
                "params": model.classifier.parameters(),
                "lr": float(t.get("classifier_lr", 1e-4)),
            },
        ],
        weight_decay=0.05,
    )

    epochs = int(t["epochs"])

    sch = optim.lr_scheduler.CosineAnnealingLR(
        opt,
        T_max=epochs,
        eta_min=1e-6,
    )

    bp = out / "best_stage3.pth"
    boundary = FinalTestBoundary(bp)

    for ep in range(epochs):

        model.train()

        for batch in tr:

            eeg, y = (batch[0], batch[1]) if len(batch) == 2 else (batch[0], batch[3])

            eeg = eeg.to(device)
            y = y.to(device)

            opt.zero_grad(
                set_to_none=True
            )

            z = model(
                eeg,
                y,
                mask_ratio=0.0,
            )

            z["loss"].backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                1.0,
            )

            opt.step()

        sch.step()

        m = eval_stage3(
            model,
            va,
            device,
        )

        if boundary.consider_validation(m["balanced_accuracy"], ep + 1):
            torch.save(
                model.state_dict(),
                bp
            )

        if (
            ep == 0
            or (ep + 1) % 10 == 0
        ):
            print(
                ep + 1,
                m,
                "best",
                boundary.best_validation_balanced_accuracy,
                flush=True,
            )

    boundary.freeze_selection()
    model.load_state_dict(
        torch.load(
            bp,
            map_location=device
        )
    )
    boundary.mark_selected_checkpoint_reloaded(bp)

    # Do not construct the final-test partition before validation-only
    # checkpoint selection has finished and the selected state is reloaded.
    boundary.authorize_final_test_access()
    if eeg_only:
        test_idx = load_task_role_indices(t, "final_test")
        if (
            np.intersect1d(train_idx, test_idx).size
            or np.intersect1d(validation_idx, test_idx).size
        ):
            raise AssertionError("Final-test overlap")
        ted = EEGIndexedDataset(eeg_path, label_path, test_idx, "final_test")
        te = DataLoader(ted, batch_size=64, shuffle=False)
    else:
        ted = load_subject_test(t["subject"], t["paradigm"], fold)
        te = DataLoader(BCIDataset(ted), batch_size=64, shuffle=False)

    if t.get("evaluate_zero_shot", False):
        zero_shot = EEGZeroShotClassifier(str(ckpt), 10).to(device)
        test, zero_test = eval_stage3_pair(model, zero_shot, te, device)
    else:
        test = eval_stage3(model, te, device)
        zero_test = None

    save_json(
        out / "result.json",
        {
            **t,
            "n_train": len(trd) if eeg_only else len(trd["labels"]),
            "n_val": len(vad) if eeg_only else len(vad["labels"]),
            "n_test": len(ted) if eeg_only else len(ted["labels"]),
            "best_val_balanced_accuracy": boundary.best_validation_balanced_accuracy,
            "best_epoch": boundary.best_epoch,
            "final_test_boundary": boundary.audit(),
            "final_test_loader_traversals": 1,
            **{
                f"test_{k}": v
                for k, v in test.items()
            },
            **({f"zero_shot_test_{k}": v for k, v in zero_test.items()} if zero_test else {}),
        },
    )


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--kind",
        choices=[
            "stage1",
            "stage2",
            "stage3",
        ],
        required=True,
    )

    ap.add_argument(
        "--manifest",
        required=True,
    )

    ap.add_argument(
        "--index",
        type=int,
        required=True,
    )

    a = ap.parse_args()

    task = read_jsonl(
        a.manifest,
        a.index,
    )

    {
        "stage1": stage1,
        "stage2": stage2,
        "stage3": stage3,
    }[a.kind](task)


if __name__ == "__main__":
    main()
