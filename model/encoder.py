import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvolutionModule(nn.Module):
    """Convolution module for Conformer"""
    def __init__(self, d_model, kernel_size=31, dropout=0.1):
        super().__init__()
        self.layer_norm = nn.LayerNorm(d_model)
        self.pointwise_conv1 = nn.Conv1d(d_model, d_model * 2, kernel_size=1)
        self.glu = nn.GLU(dim=1)
        self.depthwise_conv = nn.Conv1d(
            d_model, d_model, kernel_size,
            padding=(kernel_size - 1) // 2, groups=d_model
        )
        self.batch_norm = nn.BatchNorm1d(d_model)
        self.pointwise_conv2 = nn.Conv1d(d_model, d_model, kernel_size=1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: (batch, time, d_model)
        x = self.layer_norm(x)
        x = x.transpose(1, 2)  # (batch, d_model, time)
        x = self.pointwise_conv1(x)
        x = self.glu(x)
        x = self.depthwise_conv(x)
        x = self.batch_norm(x)
        x = F.silu(x)
        x = self.pointwise_conv2(x)
        x = self.dropout(x)
        return x.transpose(1, 2)

class FeedForwardModule(nn.Module):
    """Feed forward module with Swish activation"""
    def __init__(self, d_model, expansion_factor=4, dropout=0.1):
        super().__init__()
        self.layer_norm = nn.LayerNorm(d_model)
        self.linear1 = nn.Linear(d_model, d_model * expansion_factor)
        self.linear2 = nn.Linear(d_model * expansion_factor, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.layer_norm(x)
        x = self.linear1(x)
        x = F.silu(x) 
        x = self.dropout(x)
        x = self.linear2(x)
        return self.dropout(x)

class MultiHeadedSelfAttention(nn.Module):
    """Multi-headed self-attention (Standard, but used with LayerNorm elsewhere)"""
    def __init__(self, d_model, nhead, dropout=0.1):
        super().__init__()
        self.layer_norm = nn.LayerNorm(d_model)
        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.nhead = nhead
        self.head_dim = d_model // nhead

    def forward(self, x, mask=None):
        x = self.layer_norm(x)
        batch, seq_len, _ = x.shape
        q = self.query(x).view(batch, seq_len, self.nhead, self.head_dim).transpose(1, 2)
        k = self.key(x).view(batch, seq_len, self.nhead, self.head_dim).transpose(1, 2)
        v = self.value(x).view(batch, seq_len, self.nhead, self.head_dim).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        context = torch.matmul(attn, v).transpose(1, 2).reshape(batch, seq_len, -1)
        return self.out_proj(context)

class ConformerBlock(nn.Module):
    """Single Conformer block using Macaron style"""
    def __init__(self, d_model, nhead, kernel_size=31, dropout=0.1, ff_expansion=4):
        super().__init__()
        self.ff1 = FeedForwardModule(d_model, ff_expansion, dropout)
        self.self_attn = MultiHeadedSelfAttention(d_model, nhead, dropout)
        self.conv_module = ConvolutionModule(d_model, kernel_size, dropout)
        self.ff2 = FeedForwardModule(d_model, ff_expansion, dropout)
        self.final_norm = nn.LayerNorm(d_model)

    def forward(self, x, mask=None):
        x = x + 0.5 * self.ff1(x)
        x = x + self.self_attn(x, mask)
        x = x + self.conv_module(x)
        x = x + 0.5 * self.ff2(x)
        return self.final_norm(x)

class ConformerEncoder(nn.Module):
    def __init__(self, input_dim=80, d_model=256, nhead=4, num_layers=16,
                 kernel_size=31, dropout=0.1, ff_expansion=4, num_subsample=2):
        super().__init__()
        self.num_subsample = num_subsample
        # Stack of stride-2 Conv1d layers: total downsampling = 2^num_subsample
        layers = [nn.Conv1d(input_dim, d_model, kernel_size=3, stride=2, padding=1)]
        for _ in range(num_subsample - 1):
            layers += [nn.ReLU(), nn.Conv1d(d_model, d_model, kernel_size=3, stride=2, padding=1)]
        self.subsample = nn.Sequential(*layers)
        self.pos_encoding = nn.Parameter(torch.zeros(1, 5000, d_model))
        self.layers = nn.ModuleList([
            ConformerBlock(d_model, nhead, kernel_size, dropout, ff_expansion)
            for _ in range(num_layers)
        ])

    def forward(self, x, mask=None):
        x = x.transpose(1, 2)
        x = self.subsample(x).transpose(1, 2)
        x = x + self.pos_encoding[:, :x.size(1), :]
        for layer in self.layers:
            x = layer(x, mask)
        return x

class ConformerCTC(nn.Module):
    def __init__(self, input_dim=80, d_model=256, nhead=4, num_layers=16,
                 kernel_size=31, vocab_size=32, dropout=0.1, decoder=None,
                 num_subsample=2):
        super().__init__()
        self.encoder = ConformerEncoder(input_dim, d_model, nhead, num_layers,
                                        kernel_size, dropout, num_subsample=num_subsample)
        self.ctc_head = nn.Linear(d_model, vocab_size)
        self.decoder = decoder

    def forward(self, x, mask=None):
        encoder_out = self.encoder(x, mask)
        return self.ctc_head(encoder_out)

    def get_encoder_lengths(self, input_lengths):
        """Lengths after num_subsample stride-2 conv layers (kernel=3, pad=1)."""
        l = input_lengths
        for _ in range(self.encoder.num_subsample):
            l = (l + 2*1 - 3) // 2 + 1
        return l

    def get_log_probs(self, x, mask=None):
        return F.log_softmax(self.forward(x, mask), dim=-1)

    def decode(self, log_probs, lengths):
        if self.decoder is None:
            raise RuntimeError("No decoder set.")
        return self.decoder.decode(log_probs, lengths)