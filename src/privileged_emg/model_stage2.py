import torch
import torch.nn as nn
import torch.nn.functional as F
from .model_MAE import MAE_Encoder, MAE_Decoder

EMG_SEQ_LEN = 1000
EMG_PATCH_SIZE = 25


class HierarchicalDistillLoss(nn.Module):
    """Per-sample cosine distillation without batch-negative dependence."""
    def __init__(self, weak_weight=0.3):
        super().__init__()
        self.weak_weight = weak_weight

    def forward(self, z_eeg, z_emg, labels):
        z_eeg = F.normalize(z_eeg, dim=-1)
        z_emg = F.normalize(z_emg, dim=-1)
        B = z_eeg.size(0)
        target = z_emg.clone()
        for i in range(B):
            same = (labels == labels[i]).clone()
            same[i] = False
            if same.any():
                target[i] = target[i] + self.weak_weight * z_emg[same].mean(0)
        target = F.normalize(target, dim=-1)
        return (1.0 - (z_eeg * target).sum(dim=-1)).mean()


class EEGMAEAlignerPretrainer(nn.Module):
    """Stage 2 EEG MAE pretraining with hierarchical cosine distillation.

    The EEG encoder uses patch size 16 and sequence length 256. The frozen
    Stage 1 EMG teacher uses patch size 25 and sequence length 1000.
    """
    def __init__(
        self,
        stage1_ckpt_path,
        shared_dim=256,
        num_classes=10,
        eeg_mask_ratio=0.75,
        weak_weight=0.3,
    ):
        super().__init__()
        self.eeg_mask_ratio    = eeg_mask_ratio
        self._train_mask_ratio = eeg_mask_ratio

        self.eeg_encoder = MAE_Encoder(
            in_channels=60, embed_dim=shared_dim, patch_size=16, seq_len=256
        )
        self.eeg_decoder = MAE_Decoder(
            in_channels=60, embed_dim=shared_dim, patch_size=16, seq_len=256, is_eeg=True
        )
        self.eeg_proj = nn.Sequential(
            nn.Linear(shared_dim, shared_dim),
            nn.GELU(),
            nn.Linear(shared_dim, shared_dim),
        )
        self.classifier = nn.Sequential(
            nn.LayerNorm(shared_dim),
            nn.Dropout(0.3),
            nn.Linear(shared_dim, num_classes),
        )

        # Frozen raw-EMG encoder: patch size 25, sequence length 1000.
        self.emg_encoder = MAE_Encoder(
            in_channels=6, embed_dim=shared_dim, patch_size=EMG_PATCH_SIZE, seq_len=EMG_SEQ_LEN
        )
        self._load_emg_encoder(stage1_ckpt_path)
        for p in self.emg_encoder.parameters():
            p.requires_grad = False

        self.distill_loss = HierarchicalDistillLoss(weak_weight=weak_weight)
        self.ce_loss      = nn.CrossEntropyLoss(label_smoothing=0.1)

    def train(self, mode=True):
        super().train(mode)
        # Keep the frozen teacher in evaluation mode so dropout does not
        # change the distillation target for the same EMG sample.
        self.emg_encoder.eval()
        return self


    def _load_emg_encoder(self, ckpt_path):
        state = torch.load(ckpt_path, map_location='cpu')
        state = {k.replace('module.', '', 1): v for k, v in state.items()}
        enc_state = {k[len('emg_encoder.'):]: v
                     for k, v in state.items() if k.startswith('emg_encoder.')}
        missing, _ = self.emg_encoder.load_state_dict(enc_state, strict=True)
        print(f'[Stage 2] Loaded EMG encoder: {ckpt_path}')
        if missing:
            print(f'  [WARN] Missing keys: {missing}')

    def _mae_loss(self, orig, pred, mask, patch_size):
        target_len = pred.shape[-1]
        orig = orig[:, :, :target_len]
        loss = ((pred - orig) ** 2).mean(dim=1)
        mask_exp = mask.repeat_interleave(patch_size, dim=1)
        return (loss * mask_exp).sum() / (mask_exp.sum() + 1e-8)

    def forward(self, eeg, emg, labels, use_distill=True):
        eeg = (eeg - eeg.mean(dim=-1, keepdim=True)) / (
            eeg.std(dim=-1, keepdim=True) + 1e-8)

        vis_tokens, eeg_mask, ids_restore = self.eeg_encoder(eeg, self.eeg_mask_ratio)
        eeg_rec  = self.eeg_decoder(vis_tokens, ids_restore)
        loss_mae = self._mae_loss(eeg, eeg_rec, eeg_mask, patch_size=16)

        cls     = vis_tokens[:, 0, :]
        z_eeg   = self.eeg_proj(cls)
        logits  = self.classifier(cls)
        loss_ce = self.ce_loss(logits, labels)

        if not use_distill:
            return {
                'loss':         loss_mae,
                'loss_mae':     loss_mae,
                'loss_distill': torch.zeros(1, device=eeg.device),
                'loss_ce':      loss_ce,
                'logits':       logits,
            }

        with torch.no_grad():
            emg_n = (emg - emg.mean(dim=-1, keepdim=True)) / (
                emg.std(dim=-1, keepdim=True) + 1e-8)
            emg_tokens, _, _ = self.emg_encoder(emg_n, mask_ratio=0.0)
            z_emg = emg_tokens[:, 0, :]

        loss_distill = self.distill_loss(z_eeg, z_emg, labels)
        loss_total   = loss_mae + loss_distill + 0.1 * loss_ce

        return {
            'loss':         loss_total,
            'loss_mae':     loss_mae,
            'loss_distill': loss_distill,
            'loss_ce':      loss_ce,
            'logits':       logits,
        }


class NoDistillPretrainer(nn.Module):
    """EEG-only Stage-2 baseline with Full-identical trainable initialization."""
    def __init__(self, shared_dim=256, num_classes=10, eeg_mask_ratio=0.75):
        super().__init__()
        self.eeg_mask_ratio = eeg_mask_ratio
        self.eeg_encoder = MAE_Encoder(
            in_channels=60, embed_dim=shared_dim, patch_size=16, seq_len=256
        )
        self.eeg_decoder = MAE_Decoder(
            in_channels=60, embed_dim=shared_dim, patch_size=16,
            seq_len=256, is_eeg=True,
        )
        self.eeg_proj = nn.Sequential(
            nn.Linear(shared_dim, shared_dim), nn.GELU(),
            nn.Linear(shared_dim, shared_dim),
        )
        self.classifier = nn.Sequential(
            nn.LayerNorm(shared_dim), nn.Dropout(0.3),
            nn.Linear(shared_dim, num_classes),
        )
        self.ce_loss = nn.CrossEntropyLoss(label_smoothing=0.1)

    def forward(self, eeg, labels):
        eeg = (eeg - eeg.mean(dim=-1, keepdim=True)) / (
            eeg.std(dim=-1, keepdim=True) + 1e-8
        )
        tokens, mask, ids_restore = self.eeg_encoder(eeg, self.eeg_mask_ratio)
        reconstruction = self.eeg_decoder(tokens, ids_restore)
        target = eeg[:, :, :reconstruction.shape[-1]]
        loss = ((reconstruction - target) ** 2).mean(dim=1)
        expanded = mask.repeat_interleave(16, dim=1)
        loss_mae = (loss * expanded).sum() / (expanded.sum() + 1e-8)
        logits = self.classifier(tokens[:, 0, :])
        return {
            "loss_mae": loss_mae,
            "loss_ce": self.ce_loss(logits, labels),
            "logits": logits,
        }
