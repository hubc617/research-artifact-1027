import torch
import torch.nn as nn
from .model_MAE import MAE_Encoder

EMG_SEQ_LEN   = 1000
EMG_PATCH_SIZE = 25


class EEGDownstreamClassifier(nn.Module):
    """Stage 3 EEG-only downstream classifier."""
    def __init__(self, stage2_ckpt_path, num_classes=10, freeze_encoder=False):
        super().__init__()
        self.eeg_encoder = MAE_Encoder(
            in_channels=60, embed_dim=256, patch_size=16, seq_len=256
        )
        self._load_encoder(stage2_ckpt_path)
        if freeze_encoder:
            for p in self.eeg_encoder.parameters():
                p.requires_grad = False

        self.classifier = nn.Sequential(
            nn.LayerNorm(256),
            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )
        self.ce_loss = nn.CrossEntropyLoss(label_smoothing=0.1)

    def _load_encoder(self, ckpt_path):
        state = torch.load(ckpt_path, map_location='cpu')
        state = {k.replace('module.', '', 1): v for k, v in state.items()}
        enc_state = {k[len('eeg_encoder.'):]: v
                     for k, v in state.items() if k.startswith('eeg_encoder.')}
        missing, _ = self.eeg_encoder.load_state_dict(enc_state, strict=True)
        print(f'[Stage 3] Loaded EEG encoder: {ckpt_path}')
        if missing:
            print(f'  [WARN] Missing keys: {missing}')

    def forward(self, eeg, labels=None, mask_ratio=0.0):
        eeg = (eeg - eeg.mean(dim=-1, keepdim=True)) / (
            eeg.std(dim=-1, keepdim=True) + 1e-8)
        tokens, _, _ = self.eeg_encoder(eeg, mask_ratio=mask_ratio)
        logits = self.classifier(tokens[:, 0, :])
        if labels is not None:
            return {'loss': self.ce_loss(logits, labels), 'logits': logits}
        return {'logits': logits}


class EEGZeroShotClassifier(nn.Module):
    """Stage-2 auxiliary EEG classifier evaluated without target fine-tuning."""
    def __init__(self, stage2_ckpt_path, num_classes=10):
        super().__init__()
        self.eeg_encoder = MAE_Encoder(
            in_channels=60, embed_dim=256, patch_size=16, seq_len=256
        )
        self.classifier = nn.Sequential(
            nn.LayerNorm(256), nn.Dropout(0.3), nn.Linear(256, num_classes)
        )
        state = torch.load(stage2_ckpt_path, map_location="cpu")
        state = {key.removeprefix("module."): value for key, value in state.items()}
        encoder = {
            key.removeprefix("eeg_encoder."): value
            for key, value in state.items() if key.startswith("eeg_encoder.")
        }
        classifier = {
            key.removeprefix("classifier."): value
            for key, value in state.items() if key.startswith("classifier.")
        }
        self.eeg_encoder.load_state_dict(encoder, strict=True)
        self.classifier.load_state_dict(classifier, strict=True)

    def forward(self, eeg):
        eeg = (eeg - eeg.mean(dim=-1, keepdim=True)) / (
            eeg.std(dim=-1, keepdim=True) + 1e-8
        )
        tokens, _, _ = self.eeg_encoder(eeg, mask_ratio=0.0)
        return {"logits": self.classifier(tokens[:, 0, :])}


class EMGDownstreamClassifier(nn.Module):
    """Stage 3 EMG-only downstream classifier."""
    def __init__(self, stage1_ckpt_path, num_classes=10, freeze_encoder=False):
        super().__init__()
        # Match the EMG encoder configuration used by the bimodal model.
        self.emg_encoder = MAE_Encoder(
            in_channels=6, embed_dim=256,
            patch_size=EMG_PATCH_SIZE, seq_len=EMG_SEQ_LEN
        )
        self._load_encoder(stage1_ckpt_path)
        if freeze_encoder:
            for p in self.emg_encoder.parameters():
                p.requires_grad = False

        # Match the 256-dimensional EEG classifier head.
        self.classifier = nn.Sequential(
            nn.LayerNorm(256),
            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )
        self.ce_loss = nn.CrossEntropyLoss(label_smoothing=0.1)

    def _load_encoder(self, ckpt_path):
        state = torch.load(ckpt_path, map_location='cpu')
        state = {k.replace('module.', '', 1): v for k, v in state.items()}
        # Stage 1 checkpoint keys use the ``emg_encoder.`` prefix.
        enc_state = {k[len('emg_encoder.'):]: v
                     for k, v in state.items() if k.startswith('emg_encoder.')}
        missing, _ = self.emg_encoder.load_state_dict(enc_state, strict=True)
        print(f'[Stage 3] Loaded EMG encoder: {ckpt_path}')
        if missing:
            print(f'  [WARN] Missing keys: {missing}')

    def forward(self, emg, labels=None, mask_ratio=0.0):
        # Use the same EMG normalization as the bimodal branch.
        emg = (emg - emg.mean(dim=-1, keepdim=True)) / (
            emg.std(dim=-1, keepdim=True) + 1e-8)
        tokens, _, _ = self.emg_encoder(emg, mask_ratio=mask_ratio)
        logits = self.classifier(tokens[:, 0, :])   # CLS token
        if labels is not None:
            return {'loss': self.ce_loss(logits, labels), 'logits': logits}
        return {'logits': logits}


class BimodalEEGEMGClassifier(nn.Module):
    """Stage 3 EEG+EMG downstream classifier."""
    def __init__(self, stage2_ckpt_path, stage1_ckpt_path, num_classes=10):
        super().__init__()
        self.eeg_encoder = MAE_Encoder(
            in_channels=60, embed_dim=256, patch_size=16, seq_len=256
        )
        self.emg_encoder = MAE_Encoder(
            in_channels=6, embed_dim=256,
            patch_size=EMG_PATCH_SIZE, seq_len=EMG_SEQ_LEN
        )
        self._load_encoder(self.eeg_encoder, stage2_ckpt_path, 'eeg_encoder', 'Stage2 EEG')
        self._load_encoder(self.emg_encoder, stage1_ckpt_path, 'emg_encoder', 'Stage1 EMG')

        self.classifier = nn.Sequential(
            nn.LayerNorm(512),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )
        self.ce_loss = nn.CrossEntropyLoss(label_smoothing=0.1)

    def _load_encoder(self, encoder, ckpt_path, prefix, tag):
        state = torch.load(ckpt_path, map_location='cpu')
        state = {k.replace('module.', '', 1): v for k, v in state.items()}
        enc_state = {k[len(f'{prefix}.'):]: v
                     for k, v in state.items() if k.startswith(f'{prefix}.')}
        missing, _ = encoder.load_state_dict(enc_state, strict=True)
        print(f'[Stage 3 bimodal] Loaded {tag} encoder: {ckpt_path}')
        if missing:
            print(f'  [WARN] Missing keys: {missing}')

    def forward(self, eeg, emg, labels=None, eeg_mask_ratio=0.0):
        eeg = (eeg - eeg.mean(dim=-1, keepdim=True)) / (eeg.std(dim=-1, keepdim=True) + 1e-8)
        emg = (emg - emg.mean(dim=-1, keepdim=True)) / (emg.std(dim=-1, keepdim=True) + 1e-8)
        eeg_cls = self.eeg_encoder(eeg, mask_ratio=eeg_mask_ratio)[0][:, 0, :]
        emg_cls = self.emg_encoder(emg, mask_ratio=0.0)[0][:, 0, :]
        logits  = self.classifier(torch.cat([eeg_cls, emg_cls], dim=-1))
        if labels is not None:
            return {'loss': self.ce_loss(logits, labels), 'logits': logits}
        return {'logits': logits}
