import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class ModernTCNBlock(nn.Module):
    """Depthwise temporal convolution followed by pointwise channel mixing."""
    def __init__(self, dim, kernel_size=7):
        super().__init__()
        # Depthwise temporal convolution.
        self.dwconv = nn.Conv1d(dim, dim, kernel_size=kernel_size, padding=kernel_size//2, groups=dim)
        self.norm = nn.LayerNorm(dim)
        # Pointwise channel mixing with expansion and projection.
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)

    def forward(self, x):
        # x shape: (B, C, L)
        residual = x
        x = self.dwconv(x)
        x = x.transpose(1, 2)  # (B, L, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        x = x.transpose(1, 2)  # (B, C, L)
        return residual + x

class InfoNCELoss(nn.Module):
    """Symmetric temperature-scaled contrastive loss."""
    def __init__(self, temperature=0.1):
        super().__init__()
        self.temperature = temperature

    def forward(self, features1, features2):
        # L2-normalize features.
        f1 = F.normalize(features1, dim=-1)
        f2 = F.normalize(features2, dim=-1)

        # Compute the similarity matrix.
        logits = torch.matmul(f1, f2.T) / self.temperature

        # Diagonal entries are matching samples.
        labels = torch.arange(logits.size(0), device=logits.device)

        # Symmetric loss in both directions.
        loss_a = F.cross_entropy(logits, labels)
        loss_b = F.cross_entropy(logits.T, labels)
        return (loss_a + loss_b) / 2

class MAE_Encoder(nn.Module):
    def __init__(self, in_channels, embed_dim=256, patch_size=16, seq_len=256, num_heads=8, depth=4):
        super().__init__()
        self.patch_size = patch_size
        self.num_patches = seq_len // patch_size
        self.embed_dim = embed_dim

        # Patch embedding with a strided convolution.
        self.patch_embed = nn.Conv1d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

        # Local temporal processing.
        self.tcn_block = ModernTCNBlock(embed_dim, kernel_size=7)

        # Positional embeddings and CLS token.
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, embed_dim))

        # Transformer encoder.
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dim_feedforward=embed_dim*4, batch_first=True, activation='gelu')
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)

    def random_masking(self, x, mask_ratio):
        """Apply random patch masking."""
        B, L, D = x.shape
        len_keep = int(L * (1 - mask_ratio))

        # Sort random noise to obtain shuffle and restore indices.
        noise = torch.rand(B, L, device=x.device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # Keep the visible tokens.
        ids_keep = ids_shuffle[:, :len_keep]
        x_kept = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).expand(-1, -1, D))

        # Build a mask for loss computation over masked patches.
        mask = torch.ones([B, L], device=x.device)
        mask[:, :len_keep] = 0
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_kept, mask, ids_restore

    def forward(self, x, mask_ratio=0.75):
        B = x.shape[0]
        # (B, C, L) -> (B, embed_dim, num_patches)
        x = self.patch_embed(x)
        x = self.tcn_block(x)
        x = x.transpose(1, 2) # (B, num_patches, embed_dim)

        # Add positional embeddings excluding CLS.
        x = x + self.pos_embed[:, 1:, :]

        # Apply masking.
        x_kept, mask, ids_restore = self.random_masking(x, mask_ratio)

        # Prepend the CLS token.
        cls_token = self.cls_token.expand(B, -1, -1)
        cls_token = cls_token + self.pos_embed[:, :1, :]
        x_input = torch.cat((cls_token, x_kept), dim=1)

        # Encode visible tokens.
        x_encoded = self.transformer(x_input)

        # Return encoded tokens, mask, and restore indices.
        return x_encoded, mask, ids_restore

class MAE_Decoder(nn.Module):
    def __init__(self, in_channels, embed_dim=256, patch_size=16, seq_len=256, num_heads=4, depth=2, is_eeg=False):
        super().__init__()
        self.num_patches = seq_len // patch_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.is_eeg = is_eeg

        # Mask token and positional embeddings.
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, embed_dim))

        # Transformer decoder.
        decoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dim_feedforward=embed_dim*4, batch_first=True, activation='gelu')
        self.transformer = nn.TransformerEncoder(decoder_layer, num_layers=depth)

        # Reconstruction projection to C * patch_size.
        self.linear_pred = nn.Linear(embed_dim, in_channels * patch_size)

        # Optional EEG smoothing block.
        if self.is_eeg:
            self.smooth_block = nn.Sequential(
                nn.ConvTranspose1d(in_channels, in_channels, kernel_size=4, stride=1, padding=1),
                nn.GELU(),
                nn.Conv1d(in_channels, in_channels, kernel_size=3, padding=1)
            )

    def forward(self, x, ids_restore):
        B = x.shape[0]
        # Separate CLS and visible patches.
        cls_token = x[:, :1, :]
        x_kept = x[:, 1:, :]

        # Fill masked positions with mask tokens.
        mask_tokens = self.mask_token.expand(B, self.num_patches - x_kept.shape[1], -1)
        x_ = torch.cat([x_kept, mask_tokens], dim=1)

        # Restore original patch order.
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).expand(-1, -1, x.shape[2]))

        # Concatenate CLS and add positional embeddings.
        x_input = torch.cat([cls_token, x_], dim=1)
        x_input = x_input + self.pos_embed

        # Decode tokens.
        x_decoded = self.transformer(x_input)

        # Remove CLS token.
        x_decoded = x_decoded[:, 1:, :]

        # Project to reconstruction patches.
        pred = self.linear_pred(x_decoded)  # (B, num_patches, C * patch_size)

        # Fold patches into the time-domain signal.
        pred = pred.view(B, self.num_patches, self.in_channels, self.patch_size)
        pred = pred.transpose(1, 2).contiguous().view(B, self.in_channels, -1)  # (B, C, L)

        # Apply the optional EEG smoothing block.
        if self.is_eeg:
            pred = self.smooth_block(pred)
            # Match the encoder's patch-aligned target length.
            target_len = self.num_patches * self.patch_size
            if pred.shape[-1] != target_len:
                pred = F.interpolate(pred, size=target_len, mode='linear', align_corners=False)

        return pred
