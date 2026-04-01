import torch.nn as nn
import torch.nn.functional as F
import math
import torch


class ConvolutionModule(nn.Module):
    """Convolution module for Conformer"""
    def __init__(self, d_model, kernel_size=31, dropout=0.1):
        super().__init__()
        self.pointwise_conv1 = nn.Conv1d(d_model, d_model*2, kernel_size=1)
        self.glu = nn.GLU(dim=1)
        self.depthwise_conv = nn.Conv1d(
            d_model, d_model, kernel_size, 
            padding=kernel_size//2, groups=d_model
        )
        self.batch_norm = nn.BatchNorm1d(d_model)
        self.pointwise_conv2 = nn.Conv1d(d_model, d_model, kernel_size=1)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # x: (batch, time, d_model)
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
        self.linear1 = nn.Linear(d_model, d_model * expansion_factor)
        self.linear2 = nn.Linear(d_model * expansion_factor, d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        x = self.linear1(x)
        x = F.silu(x)  # Swish activation
        x = self.dropout(x)
        x = self.linear2(x)
        return self.dropout(x)

class MultiHeadedSelfAttention(nn.Module):
    """Multi-headed self-attention with relative positional encoding"""
    def __init__(self, d_model, nhead, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead
        assert self.head_dim * nhead == d_model
        
        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        batch, seq_len, _ = x.shape
        
        # Linear projections and reshape for multi-head
        q = self.query(x).view(batch, seq_len, self.nhead, self.head_dim).transpose(1, 2)
        k = self.key(x).view(batch, seq_len, self.nhead, self.head_dim).transpose(1, 2)
        v = self.value(x).view(batch, seq_len, self.nhead, self.head_dim).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        context = torch.matmul(attn_weights, v)
        context = context.transpose(1, 2).contiguous().view(batch, seq_len, self.d_model)
        
        return self.out_proj(context)

class ConformerBlock(nn.Module):
    """Single Conformer block"""
    def __init__(self, d_model, nhead, kernel_size=31, dropout=0.1, ff_expansion=4):
        super().__init__()
        self.ff1 = FeedForwardModule(d_model, ff_expansion, dropout)
        self.self_attn = MultiHeadedSelfAttention(d_model, nhead, dropout)
        self.conv_module = ConvolutionModule(d_model, kernel_size, dropout)
        self.ff2 = FeedForwardModule(d_model, ff_expansion, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        # Feed forward 1 (half-step)
        x = x + 0.5 * self.ff1(x)
        x = self.norm1(x)
        
        # Self-attention
        residual = x
        x = self.norm2(x)
        x = self.self_attn(x, mask)
        x = self.dropout(x)
        x = residual + x
        
        # Convolution
        residual = x
        x = self.norm3(x)
        x = self.conv_module(x)
        x = residual + x
        
        # Feed forward 2 (half-step)
        x = x + 0.5 * self.ff2(x)
        
        return x
    
class ConformerEncoder(nn.Module):
    """Conformer-S encoder"""
    def __init__(self, input_dim=80, d_model=256, nhead=4, num_layers=16, 
                 kernel_size=31, dropout=0.1, ff_expansion=4):
        super().__init__()
        
        # Input subsampling (optional, reduces sequence length)
        self.subsample = nn.Sequential(
            nn.Conv1d(input_dim, d_model, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv1d(d_model, d_model, kernel_size=3, stride=2, padding=1)
        )
        
        # Positional encoding
        self.pos_encoding = nn.Parameter(torch.zeros(1, 5000, d_model))
        
        # Conformer blocks
        self.layers = nn.ModuleList([
            ConformerBlock(d_model, nhead, kernel_size, dropout, ff_expansion)
            for _ in range(num_layers)
        ])
        
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        # x: (batch, time, input_dim)
        x = x.transpose(1, 2)  # (batch, input_dim, time)
        x = self.subsample(x)   # (batch, d_model, time//4)
        x = x.transpose(1, 2)   # (batch, time//4, d_model)
        
        # Add positional encoding
        x = x + self.pos_encoding[:, :x.size(1), :]
        x = self.dropout(x)
        
        # Pass through Conformer blocks
        for layer in self.layers:
            x = layer(x, mask)
        
        x = self.norm(x)
        return x
    

class ConformerCTC(nn.Module):
    """Complete Conformer-S model with CTC head"""
    def __init__(self, input_dim=80, d_model=256, nhead=4, num_layers=16,
                 kernel_size=31, vocab_size=32, dropout=0.1):
        super().__init__()
        
        self.encoder = ConformerEncoder(
            input_dim=input_dim,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            kernel_size=kernel_size,
            dropout=dropout
        )
        
        # CTC head (linear projection to vocabulary size)
        self.ctc_head = nn.Linear(d_model, vocab_size)
        
    def forward(self, x, mask=None):
        # x: (batch, time, input_dim)
        encoder_out = self.encoder(x, mask)
        logits = self.ctc_head(encoder_out)
        return logits
    
    def get_log_probs(self, x, mask=None):
        """Return log probabilities for CTC loss"""
        logits = self.forward(x, mask)
        return F.log_softmax(logits, dim=-1)
    

