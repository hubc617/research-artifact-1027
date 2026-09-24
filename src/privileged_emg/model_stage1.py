import torch
import torch.nn as nn
import torch.nn.functional as F
from .model_MAE import MAE_Encoder, MAE_Decoder, InfoNCELoss

EMG_SEQ_LEN = 1000
EMG_PATCH_SIZE = 25


class RobustEMGPretrainer(nn.Module):
    """Stage 1 EMG pretraining on mixed Overt and Silent trials.

    The EMG encoder uses patch size 25 and sequence length 1000.
    """
    def __init__(self, shared_dim=256, num_classes=10, emg_mask_ratio=0.75):
        super().__init__()
        self.emg_mask_ratio = emg_mask_ratio

        self.emg_encoder = MAE_Encoder(
            in_channels=6, embed_dim=shared_dim, patch_size=EMG_PATCH_SIZE, seq_len=EMG_SEQ_LEN
        )
        self.emg_decoder = MAE_Decoder(
            in_channels=6, embed_dim=shared_dim, patch_size=EMG_PATCH_SIZE, seq_len=EMG_SEQ_LEN, is_eeg=False
        )
        self.audio_proj = nn.Sequential(
            nn.Linear(768, 512),
            nn.GELU(),
            nn.Linear(512, shared_dim),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(shared_dim, num_classes),
        )
        self.contrastive_loss = InfoNCELoss(temperature=0.07)
        self.ce_loss = nn.CrossEntropyLoss(label_smoothing=0.1)

    def _mae_loss(self, orig, pred, mask, patch_size):
        target_len = pred.shape[-1]
        orig = orig[:, :, :target_len]
        loss = ((pred - orig) ** 2).mean(dim=1)
        mask_exp = mask.repeat_interleave(patch_size, dim=1)
        return (loss * mask_exp).sum() / (mask_exp.sum() + 1e-8)

    def forward(self, emg, audio, labels, is_overt_flag):
        emg = (emg - emg.mean(dim=-1, keepdim=True)) / (
            emg.std(dim=-1, keepdim=True) + 1e-8)

        emg_latent, emg_mask, emg_ids = self.emg_encoder(emg, self.emg_mask_ratio)
        emg_rec  = self.emg_decoder(emg_latent, emg_ids)
        loss_mae = self._mae_loss(emg, emg_rec, emg_mask, patch_size=EMG_PATCH_SIZE)

        emg_cls = emg_latent[:, 0, :]
        logits  = self.classifier(emg_cls)
        loss_ce = self.ce_loss(logits, labels)

        overt_idx = torch.where(is_overt_flag == 1)[0]
        if len(overt_idx) > 1:
            audio_global = self.audio_proj(audio[overt_idx])
            loss_align   = self.contrastive_loss(emg_cls[overt_idx], audio_global)
        else:
            loss_align = (self.audio_proj[0].weight * 0).sum() + (emg_cls * 0).sum()

        loss_total = 1.0 * loss_mae + 0.1 * loss_ce + 1.0 * loss_align
        return {
            'loss_mae':   loss_mae,
            'loss_ce':    loss_ce,
            'loss_align': loss_align,
            'loss_total': loss_total,
            'logits':     logits,
        }
