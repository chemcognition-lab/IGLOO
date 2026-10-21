import torch
import torch.nn as nn
import torch.nn.functional as F


class LinearRegressionModel(nn.Module):
    """Simple baseline: mean pooling + linear regression."""
    def __init__(self, seq_len, input_dim, final_activation=None, target_min=None, target_max=None):
        super().__init__()
        self.layernorm = nn.LayerNorm(input_dim)
        self.fc = nn.Linear(input_dim, 1)
        # Optional final activation
        if final_activation == 'tanh':
            self.final_act = nn.Tanh()
        elif final_activation == 'sigmoid':
            self.final_act = nn.Sigmoid()
        else:
            self.final_act = nn.Identity()
        self.final_activation = final_activation
        self.target_min = target_min
        self.target_max = target_max

    def forward(self, x):
        # x: (batch, seq_len, input_dim)
        pooled = torch.mean(x, dim=1)
        out = self.layernorm(pooled)
        out = self.fc(out)
        out = self.final_act(out)
        # Scale output to target range if activation is used
        if self.final_activation == 'tanh' and self.target_min is not None and self.target_max is not None:
            # Tanh outputs [-1, 1] -> scale to [target_min, target_max]
            out = 0.5 * (out + 1) * (self.target_max - self.target_min) + self.target_min
        elif self.final_activation == 'sigmoid' and self.target_min is not None and self.target_max is not None:
            # Sigmoid outputs [0, 1] -> scale to [target_min, target_max]
            out = out * (self.target_max - self.target_min) + self.target_min
        return out



class MeanPoolingModel(nn.Module):
    def __init__(self, seq_len, input_dim, hidden_dim1=1024, hidden_dim2=512, hidden_dim3=256, hidden_dim4=128, dropout_rate=0.3, final_activation=None, target_min=None, target_max=None):
        super().__init__()
        self.layernorm = nn.LayerNorm(input_dim)
        self.fc1 = nn.Linear(input_dim, hidden_dim1)
        self.gelu1 = nn.GELU()
        self.dropout1 = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.gelu2 = nn.GELU()
        self.dropout2 = nn.Dropout(dropout_rate)
        self.fc3 = nn.Linear(hidden_dim2, hidden_dim3)
        self.gelu3 = nn.GELU()
        self.dropout3 = nn.Dropout(dropout_rate)
        self.fc4 = nn.Linear(hidden_dim3, hidden_dim4)
        self.gelu4 = nn.GELU()
        self.dropout4 = nn.Dropout(dropout_rate)
        self.fc5 = nn.Linear(hidden_dim4, 1)
        # Optional final activation
        if final_activation == 'tanh':
            self.final_act = nn.Tanh()
        elif final_activation == 'sigmoid':
            self.final_act = nn.Sigmoid()
        else:
            self.final_act = nn.Identity()
        self.final_activation = final_activation
        self.target_min = target_min
        self.target_max = target_max
        # For residual connection: project out1 to hidden_dim4 if needed
        if hidden_dim1 != hidden_dim4:
            self.res_proj = nn.Linear(hidden_dim1, hidden_dim4)
        else:
            self.res_proj = nn.Identity()
    def forward(self, x):
        # x: (batch, seq_len, input_dim)
        pooled = torch.mean(x, dim=1)
        out = self.layernorm(pooled)
        out1 = self.fc1(out)
        out1 = self.gelu1(out1)
        out1 = self.dropout1(out1)
        out2 = self.fc2(out1)
        out2 = self.gelu2(out2)
        out2 = self.dropout2(out2)
        out3 = self.fc3(out2)
        out3 = self.gelu3(out3)
        out3 = self.dropout3(out3)
        out4 = self.fc4(out3)
        out4 = self.gelu4(out4)
        out4 = self.dropout4(out4)
        # Residual connection from out1 to out4
        out4 = out4 + self.res_proj(out1)
        out = self.fc5(out4)
        out = self.final_act(out)
        # Scale output to target range if activation is used
        if self.final_activation == 'tanh' and self.target_min is not None and self.target_max is not None:
            out = 0.5 * (out + 1) * (self.target_max - self.target_min) + self.target_min
        elif self.final_activation == 'sigmoid' and self.target_min is not None and self.target_max is not None:
            out = out * (self.target_max - self.target_min) + self.target_min
        return out

class AttentionConvPoolingModel(nn.Module):
    def __init__(self, seq_len, input_dim, conv_dim=256, kernel_size=5, hidden_dim1=1024, hidden_dim2=512, hidden_dim3=256, hidden_dim4=128, dropout_rate=0.3, final_activation=None, target_min=None, target_max=None):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels=input_dim, out_channels=conv_dim, kernel_size=kernel_size, padding=kernel_size//2)
        self.relu_conv = nn.ReLU()
        self.attn = nn.Linear(conv_dim, 1)
        self.layernorm = nn.LayerNorm(conv_dim)
        self.fc1 = nn.Linear(conv_dim, hidden_dim1)
        self.gelu1 = nn.GELU()
        self.dropout1 = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.gelu2 = nn.GELU()
        self.dropout2 = nn.Dropout(dropout_rate)
        self.fc3 = nn.Linear(hidden_dim2, hidden_dim3)
        self.gelu3 = nn.GELU()
        self.dropout3 = nn.Dropout(dropout_rate)
        self.fc4 = nn.Linear(hidden_dim3, hidden_dim4)
        self.gelu4 = nn.GELU()
        self.dropout4 = nn.Dropout(dropout_rate)
        self.fc5 = nn.Linear(hidden_dim4, 1)
        # Optional final activation
        if final_activation == 'tanh':
            self.final_act = nn.Tanh()
        elif final_activation == 'sigmoid':
            self.final_act = nn.Sigmoid()
        else:
            self.final_act = nn.Identity()
        self.final_activation = final_activation
        self.target_min = target_min
        self.target_max = target_max
        # For residual connection: project out1 to hidden_dim4 if needed
        if hidden_dim1 != hidden_dim4:
            self.res_proj = nn.Linear(hidden_dim1, hidden_dim4)
        else:
            self.res_proj = nn.Identity()

    def forward(self, x):
        # x: (batch, seq_len, input_dim)
        x = x.transpose(1, 2)  # (batch, input_dim, seq_len)
        x = self.conv1(x)      # (batch, conv_dim, seq_len)
        x = self.relu_conv(x)
        x = x.transpose(1, 2)  # (batch, seq_len, conv_dim)
        attn_weights = torch.softmax(self.attn(x), dim=1)
        pooled = torch.sum(attn_weights * x, dim=1)
        out = self.layernorm(pooled)
        out1 = self.fc1(out)
        out1 = self.gelu1(out1)
        out1 = self.dropout1(out1)
        out2 = self.fc2(out1)
        out2 = self.gelu2(out2)
        out2 = self.dropout2(out2)
        out3 = self.fc3(out2)
        out3 = self.gelu3(out3)
        out3 = self.dropout3(out3)
        out4 = self.fc4(out3)
        out4 = self.gelu4(out4)
        out4 = self.dropout4(out4)
        # Residual connection from out1 to out4
        out4 = out4 + self.res_proj(out1)
        out = self.fc5(out4)
        out = self.final_act(out)
        # Scale output to target range if activation is used
        if self.final_activation == 'tanh' and self.target_min is not None and self.target_max is not None:
            out = 0.5 * (out + 1) * (self.target_max - self.target_min) + self.target_min
        elif self.final_activation == 'sigmoid' and self.target_min is not None and self.target_max is not None:
            out = out * (self.target_max - self.target_min) + self.target_min
        return out


# The models below are from the FLIP repository: https://github.com/J-SNACKKB/FLIP/blob/main/baselines/models.py
# as well as Microsoft's protein sequence model repository:


class MaskedConv1d(nn.Conv1d):
    """ A masked 1-dimensional convolution layer.

    Takes the same arguments as torch.nn.Conv1D, except that the padding is set automatically.

         Shape:
            Input: (N, L, in_channels)
            input_mask: (N, L, 1), optional
            Output: (N, L, out_channels)
    """

    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size: int, stride: int=1, dilation: int=1, groups: int=1,
                 bias: bool=True):
        """
        :param in_channels: input channels
        :param out_channels: output channels
        :param kernel_size: the kernel width
        :param stride: filter shift
        :param dilation: dilation factor
        :param groups: perform depth-wise convolutions
        :param bias: adds learnable bias to output
        """
        padding = dilation * (kernel_size - 1) // 2
        super().__init__(in_channels, out_channels, kernel_size, stride=stride, dilation=dilation,
                                           groups=groups, bias=bias, padding=padding)

    def forward(self, x, input_mask=None):
        if input_mask is not None:
            x = x * input_mask
        return super().forward(x.transpose(1, 2)).transpose(1, 2)


class Attention1d(nn.Module):

    def __init__(self, in_dim):
        super().__init__()
        self.layer = MaskedConv1d(in_dim, 1, 1)

    def forward(self, x, input_mask=None):
        n, ell, _ = x.shape                    # x:    (batch, seq_len, d_embedding)
        attn = self.layer(x)                   # attn: (batch, seq_len, 1)  — scalar score per position
        attn = attn.view(n, -1)                # attn: (batch, seq_len)
        if input_mask is not None:
            attn = attn.masked_fill_(~input_mask.view(n, -1).bool(), float('-inf'))
                                               # attn: (batch, seq_len)  — masked positions → -inf
        attn = F.softmax(attn, dim=-1).view(n, -1, 1)
                                               # attn: (batch, seq_len, 1)  — softmax over positions
        out = (attn * x).sum(dim=1)            # out:  (batch, d_embedding) — weighted sum over positions
        return out


class ESMAttention1d(nn.Module):
    """Attention pooling over per-residue ESM embeddings followed by a configurable MLP."""
    def __init__(self, max_length, d_embedding, hidden_dim1=256, hidden_dim2=128,
                 hidden_dim3=64, hidden_dim4=32, dropout_rate=0.3,
                 final_activation=None):
        super(ESMAttention1d, self).__init__()
        self.attention1d = Attention1d(in_dim=d_embedding)
        self.layernorm = nn.LayerNorm(d_embedding)
        self.fc1 = nn.Linear(d_embedding, hidden_dim1)
        self.gelu1 = nn.GELU()
        self.drop1 = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.gelu2 = nn.GELU()
        self.drop2 = nn.Dropout(dropout_rate)
        self.fc3 = nn.Linear(hidden_dim2, hidden_dim3)
        self.gelu3 = nn.GELU()
        self.drop3 = nn.Dropout(dropout_rate)
        self.fc4 = nn.Linear(hidden_dim3, hidden_dim4)
        self.gelu4 = nn.GELU()
        self.drop4 = nn.Dropout(dropout_rate)
        self.final = nn.Linear(hidden_dim4, 1)
        if hidden_dim1 != hidden_dim4:
            self.res_proj = nn.Linear(hidden_dim1, hidden_dim4)
        else:
            self.res_proj = nn.Identity()
        if final_activation == 'tanh':
            self.final_act = nn.Tanh()
        elif final_activation == 'sigmoid':
            self.final_act = nn.Sigmoid()
        else:
            self.final_act = nn.Identity()

    def forward(self, x, input_mask):
        # x:         (batch, seq_len, d_embedding)
        # input_mask:(batch, seq_len)
        x = self.attention1d(x, input_mask=input_mask.unsqueeze(-1))
                                               # x:    (batch, d_embedding)   — attention-pooled
        x = self.layernorm(x)                  # x:    (batch, d_embedding)
        out1 = self.drop1(self.gelu1(self.fc1(x)))     # out1: (batch, hidden_dim1)
        out2 = self.drop2(self.gelu2(self.fc2(out1)))  # out2: (batch, hidden_dim2)
        out3 = self.drop3(self.gelu3(self.fc3(out2)))  # out3: (batch, hidden_dim3)
        out4 = self.drop4(self.gelu4(self.fc4(out3)))  # out4: (batch, hidden_dim4)
        out4 = out4 + self.res_proj(out1)      # out4: (batch, hidden_dim4)  — residual from out1
        return self.final_act(self.final(out4))# out:  (batch, 1)


class ESMAttention1dPreMLP(ESMAttention1d):
    """Same head as ESMAttention1d, but with a shared 2-layer per-RESIDUE MLP
    applied to every token embedding BEFORE the attention pool.

    Motivation: in the plain ESMAttention1d, every learnable non-linearity is
    post-pooling, so ∂F/∂x_i ≈ attn_i · (∂F/∂pooled) — the per-position IG
    pattern is set by the (shallow) attention layer and is blind to the MLP.
    Moving a learnable transform in front of the pool makes ∂F/∂x_i depend on
    the trained weights per position, so IG can 'see' the learned function.
    """
    def __init__(self, max_length, d_embedding, hidden_dim1=256, hidden_dim2=128,
                 hidden_dim3=64, hidden_dim4=32, dropout_rate=0.3,
                 final_activation=None, pre_dim=None):
        super().__init__(max_length, d_embedding, hidden_dim1, hidden_dim2,
                         hidden_dim3, hidden_dim4, dropout_rate, final_activation)
        pd_ = pre_dim or d_embedding
        self.pre = nn.Sequential(
            nn.Linear(d_embedding, pd_), nn.GELU(), nn.Dropout(dropout_rate),
            nn.Linear(pd_, d_embedding),
        )

    def forward(self, x, input_mask):
        x = x + self.pre(x)                     # residual per-residue transform
        return super().forward(x, input_mask)


class ESMAttentionChannel(nn.Module):
    """Attention pools the CHANNEL axis (960→1, same weights at every
    position) instead of the sequence-length axis, so the per-residue
    sequence dimension L survives into a small per-position MLP before the
    final (length-)attention pool.

    x: (B,L,960) → s_i = softmax_960(w)·x_i  (B,L)            [channel attn]
      → per-position MLP (shared, scalar→hidden, same depth/residual
        pattern as ESMAttention1d)                             (B,L,H)
      → attn_i = softmax_L(Linear(H→1));  f_i = Linear(H→1)(h_i)
      → fitness = Σ_i attn_i · f_i
    """
    def __init__(self, max_length, d_embedding, hidden_dim1=256, hidden_dim2=128,
                 hidden_dim3=64, hidden_dim4=32, dropout_rate=0.3,
                 final_activation=None):
        super().__init__()
        self.channel_logits = nn.Parameter(torch.zeros(d_embedding))
        self.fc1 = nn.Linear(1, hidden_dim1)
        self.gelu1 = nn.GELU()
        self.drop1 = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.gelu2 = nn.GELU()
        self.drop2 = nn.Dropout(dropout_rate)
        self.fc3 = nn.Linear(hidden_dim2, hidden_dim3)
        self.gelu3 = nn.GELU()
        self.drop3 = nn.Dropout(dropout_rate)
        self.fc4 = nn.Linear(hidden_dim3, hidden_dim4)
        self.gelu4 = nn.GELU()
        self.drop4 = nn.Dropout(dropout_rate)
        self.res_proj = (nn.Linear(hidden_dim1, hidden_dim4)
                         if hidden_dim1 != hidden_dim4 else nn.Identity())
        self.pool_score = nn.Linear(hidden_dim4, 1)
        self.out_proj = nn.Linear(hidden_dim4, 1)
        if final_activation == 'tanh':
            self.final_act = nn.Tanh()
        elif final_activation == 'sigmoid':
            self.final_act = nn.Sigmoid()
        else:
            self.final_act = nn.Identity()

    def forward(self, x, input_mask):
        ch_attn = F.softmax(self.channel_logits, dim=-1)          # (D,)
        s = (x * ch_attn).sum(dim=-1, keepdim=True)                # (B,L,1)
        h1 = self.drop1(self.gelu1(self.fc1(s)))
        h2 = self.drop2(self.gelu2(self.fc2(h1)))
        h3 = self.drop3(self.gelu3(self.fc3(h2)))
        h4 = self.drop4(self.gelu4(self.fc4(h3)))
        h4 = h4 + self.res_proj(h1)                                  # (B,L,hidden4)
        scores = self.pool_score(h4).squeeze(-1)                      # (B,L)
        scores = scores.masked_fill(~input_mask.bool(), float('-inf'))
        attn = F.softmax(scores, dim=-1).unsqueeze(-1)                  # (B,L,1)
        f = self.out_proj(h4)                                             # (B,L,1)
        out = (attn * f).sum(dim=1)                                         # (B,1)
        return self.final_act(out)


class ESMPerResidueMLP(nn.Module):
    """Moves the ENTIRE MLP (same depth/residual pattern as ESMAttention1d)
    to run per-residue (shared weights) directly on the raw d_embedding
    vector, then aggregates over positions LAST via plain mean or a learned
    attention pool (``final_pool``).

    x_i (D,) → per-position MLP (shared) → f_i (scalar)
      mean:  fitness = mean_i f_i    → ∂fitness/∂x_i = (1/L)·∂f_i/∂x_i
             (uniform weight — ALL per-position IG signal comes from the
             trained per-position MLP, not a shallow pooling layer)
      attn:  attn_i = softmax_L(Linear(h4_i));  fitness = Σ_i attn_i · f_i
    """
    def __init__(self, max_length, d_embedding, hidden_dim1=256, hidden_dim2=128,
                 hidden_dim3=64, hidden_dim4=32, dropout_rate=0.3,
                 final_activation=None, final_pool='mean'):
        super().__init__()
        assert final_pool in ('mean', 'attention'), final_pool
        self.final_pool = final_pool
        self.layernorm = nn.LayerNorm(d_embedding)
        self.fc1 = nn.Linear(d_embedding, hidden_dim1)
        self.gelu1 = nn.GELU()
        self.drop1 = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.gelu2 = nn.GELU()
        self.drop2 = nn.Dropout(dropout_rate)
        self.fc3 = nn.Linear(hidden_dim2, hidden_dim3)
        self.gelu3 = nn.GELU()
        self.drop3 = nn.Dropout(dropout_rate)
        self.fc4 = nn.Linear(hidden_dim3, hidden_dim4)
        self.gelu4 = nn.GELU()
        self.drop4 = nn.Dropout(dropout_rate)
        self.res_proj = (nn.Linear(hidden_dim1, hidden_dim4)
                         if hidden_dim1 != hidden_dim4 else nn.Identity())
        self.final = nn.Linear(hidden_dim4, 1)
        if final_pool == 'attention':
            self.pool_score = nn.Linear(hidden_dim4, 1)
        if final_activation == 'tanh':
            self.final_act = nn.Tanh()
        elif final_activation == 'sigmoid':
            self.final_act = nn.Sigmoid()
        else:
            self.final_act = nn.Identity()

    def forward(self, x, input_mask):
        x = self.layernorm(x)
        out1 = self.drop1(self.gelu1(self.fc1(x)))
        out2 = self.drop2(self.gelu2(self.fc2(out1)))
        out3 = self.drop3(self.gelu3(self.fc3(out2)))
        out4 = self.drop4(self.gelu4(self.fc4(out3)))
        out4 = out4 + self.res_proj(out1)                          # (B,L,hidden4)
        f = self.final(out4)                                        # (B,L,1)
        mask = input_mask.bool()
        if self.final_pool == 'mean':
            f = f.masked_fill(~mask.unsqueeze(-1), 0.0)
            counts = input_mask.sum(dim=1, keepdim=True).clamp(min=1)  # (B,1)
            out = f.sum(dim=1) / counts                                 # (B,1)
        else:  # attention
            scores = self.pool_score(out4).squeeze(-1)                   # (B,L)
            scores = scores.masked_fill(~mask, float('-inf'))
            attn = F.softmax(scores, dim=-1).unsqueeze(-1)                 # (B,L,1)
            out = (attn * f).sum(dim=1)                                     # (B,1)
        return self.final_act(out)


class ESMCrossAttnPerResidue(ESMPerResidueMLP):
    """Adds 1-2 real multi-head self-attention layers across POSITIONS (so
    residues exchange information — unlike every other architecture here)
    before the per-position MLP + mean pool inherited from ESMPerResidueMLP.
    Best chance of modeling true epistasis; IG attributions are harder to
    interpret cleanly since positions now mix before the deep head runs.
    """
    def __init__(self, max_length, d_embedding, hidden_dim1=256, hidden_dim2=128,
                 hidden_dim3=64, hidden_dim4=32, dropout_rate=0.3,
                 final_activation=None, n_layers=2, n_heads=8):
        super().__init__(max_length, d_embedding, hidden_dim1, hidden_dim2, hidden_dim3,
                         hidden_dim4, dropout_rate, final_activation, final_pool='mean')
        if d_embedding % n_heads != 0:
            n_heads = 1  # fall back for embedding dims that don't divide evenly (e.g. one-hot's 21)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_embedding, nhead=n_heads, dim_feedforward=d_embedding * 2,
            dropout=dropout_rate, activation='gelu', batch_first=True,
        )
        self.cross_attn = nn.TransformerEncoder(enc_layer, num_layers=n_layers)

    def forward(self, x, input_mask):
        key_padding_mask = ~input_mask.bool()          # True = ignore (padding)
        # guard fully-padded rows so softmax never sees an all -inf row
        all_pad = key_padding_mask.all(dim=1)
        if all_pad.any():
            key_padding_mask = key_padding_mask.clone()
            key_padding_mask[all_pad, 0] = False
        x = self.cross_attn(x, src_key_padding_mask=key_padding_mask)
        return super().forward(x, input_mask)


class ESMPerResidueFlatten(nn.Module):
    """Per-residue MLP (shared weights, same depth/residual pattern as
    ESMAttention1d) with NO pooling operation at all: the per-position hidden
    representations are flattened and fed directly into a single linear head,
    so the position -> prediction weights are freely learned, per-position
    coefficients -- not constrained to a mean (uniform 1/L, like
    esm_perres_mean) or a softmax-normalized convex combination (attention
    pool, like esm_perres_attn).

    Requires FIXED-length input: every batch must have exactly `max_length`
    real positions, no padding. True for this per-protein/per-landscape
    synthetic setup (point substitutions only, no indels -- every variant of
    a given protein has the same length) but does NOT generalize to
    variable-length sequences without redesign.
    """
    def __init__(self, max_length, d_embedding, hidden_dim1=256, hidden_dim2=128,
                 hidden_dim3=64, hidden_dim4=32, dropout_rate=0.3,
                 final_activation=None):
        super().__init__()
        self.L = max_length
        self.layernorm = nn.LayerNorm(d_embedding)
        self.fc1 = nn.Linear(d_embedding, hidden_dim1)
        self.gelu1 = nn.GELU()
        self.drop1 = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.gelu2 = nn.GELU()
        self.drop2 = nn.Dropout(dropout_rate)
        self.fc3 = nn.Linear(hidden_dim2, hidden_dim3)
        self.gelu3 = nn.GELU()
        self.drop3 = nn.Dropout(dropout_rate)
        self.fc4 = nn.Linear(hidden_dim3, hidden_dim4)
        self.gelu4 = nn.GELU()
        self.drop4 = nn.Dropout(dropout_rate)
        self.res_proj = (nn.Linear(hidden_dim1, hidden_dim4)
                         if hidden_dim1 != hidden_dim4 else nn.Identity())
        self.head = nn.Linear(max_length * hidden_dim4, 1)
        if final_activation == 'tanh':
            self.final_act = nn.Tanh()
        elif final_activation == 'sigmoid':
            self.final_act = nn.Sigmoid()
        else:
            self.final_act = nn.Identity()

    def forward(self, x, input_mask):
        # x: (B, L, D). No mask-based pooling here -- this architecture
        # assumes every position is real (fixed-length inputs, no padding).
        b, l, _ = x.shape
        x = self.layernorm(x)
        out1 = self.drop1(self.gelu1(self.fc1(x)))
        out2 = self.drop2(self.gelu2(self.fc2(out1)))
        out3 = self.drop3(self.gelu3(self.fc3(out2)))
        out4 = self.drop4(self.gelu4(self.fc4(out3)))
        out4 = out4 + self.res_proj(out1)                  # (B, L, hidden4)
        flat = out4.reshape(b, -1)                            # (B, L*hidden4)
        if flat.shape[1] != self.head.in_features:
            raise ValueError(
                f"ESMPerResidueFlatten got L={l} but was built for "
                f"max_length={self.L} -- this architecture requires fixed-length "
                f"input (no padding)."
            )
        out = self.head(flat)
        return self.final_act(out)


class ESMAttention1dMean(nn.Module):
    """Attention1d removed, leaving a basic linear model"""
    def __init__(self, d_embedding): # [batch x embedding (1280)]  --> [batch x 1]
        super(ESMAttention1dMean, self).__init__()
        self.linear = nn.Linear(d_embedding, d_embedding)
        self.relu = nn.ReLU()
        self.final = nn.Linear(d_embedding, 1)

    def forward(self, x):
        x = self.relu(self.linear(x))
        x = self.final(x)
        return x