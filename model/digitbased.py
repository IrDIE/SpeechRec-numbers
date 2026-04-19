import math
import torch
import torch.nn as nn
import torchaudio


def _sinusoidal_pe(max_len: int, d_model: int) -> torch.Tensor:
    pe  = torch.zeros(max_len, d_model)
    pos = torch.arange(max_len).unsqueeze(1).float()
    div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div)
    return pe  # [max_len, d_model]


class NumberModel(nn.Module):
    """
    Log-mel → Linear proj + sinusoidal PE → Conformer encoder
            → cross-attention with 6 learned digit queries
            → shared linear classifier (10 classes per digit)

    Optionally adds a mean-pooled auxiliary head that predicts the number of
    significant digits (classes: 1-6), weighted by aux_weight during training.
    Returns logits [B, n_digits, 10], or (logits, len_logits [B, n_digits])
    when aux_length=True.
    """

    def __init__(
        self,
        input_dim:   int  = 80,
        d_model:     int  = 192,
        num_layers:  int  = 4,
        num_heads:   int  = 4,
        ffn_dim:     int  = 384,
        kernel_size: int  = 31,
        n_digits:    int  = 6,
        dropout:     float = 0.1,
        aux_length:  bool = False,
    ):
        super().__init__()
        self.n_digits   = n_digits
        self.aux_length = aux_length

        self.proj = nn.Linear(input_dim, d_model)
        self.register_buffer("pe", _sinusoidal_pe(5000, d_model))

        self.encoder = torchaudio.models.Conformer(
            input_dim=d_model,
            num_heads=num_heads,
            ffn_dim=ffn_dim,
            num_layers=num_layers,
            depthwise_conv_kernel_size=kernel_size,
            dropout=dropout,
        )

        self.digit_queries = nn.Embedding(n_digits, d_model)
        self.cross_attn    = nn.MultiheadAttention(d_model, num_heads,
                                                   dropout=dropout, batch_first=True)
        self.classifier    = nn.Linear(d_model, 10)   # shared across all digit positions

        if aux_length:
            self.length_head = nn.Linear(d_model, n_digits)  # predict 1-n_digits significant digits

    def forward(self, feats: torch.Tensor, lengths: torch.Tensor):
        # feats: [B, T, input_dim]   lengths: [B] valid frame counts
        B, T, _ = feats.shape

        x = self.proj(feats) + self.pe[:T]           # [B, T, d_model]
        x, enc_lengths = self.encoder(x, lengths)    # [B, T, d_model]

        # key_padding_mask: True = padded (ignored) position
        pad_mask = (torch.arange(T, device=x.device).unsqueeze(0)
                    >= enc_lengths.unsqueeze(1))      # [B, T]

        queries  = self.digit_queries.weight.unsqueeze(0).expand(B, -1, -1)  # [B, n_digits, d_model]
        attended, _ = self.cross_attn(queries, x, x, key_padding_mask=pad_mask)  # [B, n_digits, d_model]

        logits = self.classifier(attended)            # [B, n_digits, 10]

        if self.aux_length:
            valid  = (~pad_mask).float()              # [B, T]
            pooled = (x * valid.unsqueeze(-1)).sum(1) / valid.sum(1, keepdim=True)  # [B, d_model]
            return logits, self.length_head(pooled)   # [B, n_digits, 10], [B, n_digits]

        return logits
