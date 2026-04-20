import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvolutionModule(nn.Module):
    def __init__(self, d_model, kernel_size=31, dropout=0.1):
        super().__init__()
        self.layer_norm = nn.LayerNorm(d_model)
        self.pointwise_conv1 = nn.Conv1d(d_model, d_model * 2, kernel_size=1)
        self.glu = nn.GLU(dim=1)
        self.depthwise_conv = nn.Conv1d(d_model, d_model, kernel_size,
                                         padding=(kernel_size - 1) // 2, groups=d_model)
        self.batch_norm = nn.BatchNorm1d(d_model)
        self.pointwise_conv2 = nn.Conv1d(d_model, d_model, kernel_size=1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.layer_norm(x)
        x = x.transpose(1, 2)
        x = self.glu(self.pointwise_conv1(x))
        x = F.silu(self.batch_norm(self.depthwise_conv(x)))
        x = self.dropout(self.pointwise_conv2(x))
        return x.transpose(1, 2)


class FeedForwardModule(nn.Module):
    def __init__(self, d_model, expansion_factor=4, dropout=0.1):
        super().__init__()
        self.layer_norm = nn.LayerNorm(d_model)
        self.linear1 = nn.Linear(d_model, d_model * expansion_factor)
        self.linear2 = nn.Linear(d_model * expansion_factor, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.layer_norm(x)
        return self.dropout(self.linear2(self.dropout(F.silu(self.linear1(x)))))


class ConformerBlock(nn.Module):
    """Single Conformer block (Macaron style: 0.5*FF + Attn + Conv + 0.5*FF + Norm)."""

    def __init__(self, d_model, nhead, kernel_size=31, dropout=0.1, ff_expansion=4):
        super().__init__()
        self.ff1 = FeedForwardModule(d_model, ff_expansion, dropout)
        self.attn_norm = nn.LayerNorm(d_model)
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.conv_module = ConvolutionModule(d_model, kernel_size, dropout)
        self.ff2 = FeedForwardModule(d_model, ff_expansion, dropout)
        self.final_norm = nn.LayerNorm(d_model)

    def forward(self, x):
        x = x + 0.5 * self.ff1(x)
        normed = self.attn_norm(x)
        attn_out, _ = self.self_attn(normed, normed, normed)
        x = x + attn_out
        x = x + self.conv_module(x)
        x = x + 0.5 * self.ff2(x)
        return self.final_norm(x)


class ConformerEncoder(nn.Module):
    def __init__(self, input_dim=80, d_model=256, nhead=4, num_layers=16,
                 kernel_size=31, dropout=0.1, ff_expansion=4, num_subsample=2):
        super().__init__()
        self.num_subsample = num_subsample
        layers = [nn.Conv1d(input_dim, d_model, kernel_size=3, stride=2, padding=1)]
        for _ in range(num_subsample - 1):
            layers += [nn.ReLU(), nn.Conv1d(d_model, d_model, kernel_size=3, stride=2, padding=1)]
        self.subsample = nn.Sequential(*layers)
        self.pos_encoding = nn.Parameter(torch.zeros(1, 5000, d_model))
        self.layers = nn.ModuleList([
            ConformerBlock(d_model, nhead, kernel_size, dropout, ff_expansion)
            for _ in range(num_layers)
        ])

    def forward(self, x):
        x = self.subsample(x.transpose(1, 2)).transpose(1, 2)
        x = x + self.pos_encoding[:, : x.size(1), :]
        for layer in self.layers:
            x = layer(x)
        return x


class ConformerCTC(nn.Module):
    def __init__(self, input_dim=80, d_model=256, nhead=4, num_layers=16,
                 kernel_size=31, vocab_size=32, dropout=0.1, decoder=None, num_subsample=2):
        super().__init__()
        self.encoder = ConformerEncoder(input_dim, d_model, nhead, num_layers,
                                         kernel_size, dropout, num_subsample=num_subsample)
        self.ctc_head = nn.Linear(d_model, vocab_size)
        self.decoder = decoder

    def forward(self, x):
        return self.ctc_head(self.encoder(x))

    def get_encoder_lengths(self, input_lengths):
        """Lengths after num_subsample stride-2 conv layers (kernel=3, pad=1)."""
        l = input_lengths
        for _ in range(self.encoder.num_subsample):
            l = (l + 2 * 1 - 3) // 2 + 1
        return l

    def get_log_probs(self, x):
        return F.log_softmax(self.forward(x), dim=-1)

    def decode(self, log_probs, lengths):
        if self.decoder is None:
            raise RuntimeError("No decoder set.")
        return self.decoder.decode(log_probs, lengths)
