import torch
import torch.nn as nn
from torch.nn import functional as F
import math


def safe_l2_normalize(x: torch.Tensor, dim: int = -1, eps: float = 1e-6) -> torch.Tensor:
    """Normalize in FP32 so mixed precision cannot turn epsilon into zero."""
    with torch.autocast(device_type="cuda", enabled=False):
        return F.normalize(x.float(), p=2, dim=dim, eps=eps)


class GPTCConfig:
    """ base GPT config, params common to all GPT versions """
    
    embd_pdrop: float = 0.1
    resid_pdrop: float = 0.1
    attn_pdrop: float = 0.1
    max_seq_len: int = 256
    n_ind: int = 256
    n_embd: int = 256
    n_head: int = 16
    n_layer: int = 24
    detach_x: bool = False
    detach_target: bool = True
    l2_normalized: bool = True
    n_classes: int = -1
    fully_separated: bool = False
    def __init__(self, vocab_size, block_size, **kwargs):
        self.vocab_size = vocab_size
        self.block_size = block_size
        for k,v in kwargs.items():
            setattr(self, k, v)


class CausalSelfAttention(nn.Module):
    """
    A vanilla multi-head masked self-attention layer with a projection at the end.
    It is possible to use torch.nn.MultiheadAttention here but I am including an
    explicit implementation here to show that there is nothing too scary here.
    """

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads
        self.key = nn.Linear(config.n_embd, config.n_embd)
        self.query = nn.Linear(config.n_embd, config.n_embd)
        self.value = nn.Linear(config.n_embd, config.n_embd)
        # regularization
        self.attn_drop = nn.Dropout(config.attn_pdrop)
        self.resid_drop = nn.Dropout(config.resid_pdrop)
        # output projection
        self.proj = nn.Linear(config.n_embd, config.n_embd)

        self.n_head = config.n_head

        self.p_attn_drop = config.attn_pdrop
        mask = torch.tril(torch.ones(config.block_size,
                                     config.block_size))
        self.register_buffer("mask", mask.view(1, 1, config.block_size, config.block_size))

    def forward(self, x, layer_past=None):
        B, T, C = x.size()
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        k = self.key(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        present = torch.stack((k, v)) # (2, B, nh, T, hs)
        if layer_past is not None:
            past_key, past_value = layer_past
            k = torch.cat((past_key, k), dim=-2)
            v = torch.cat((past_value, v), dim=-2)
        
        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        if layer_past is None:
            att = att.masked_fill(self.mask[:,:,:T,:T] == 0, float('-inf'))

        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_drop(self.proj(y))
        return y, present   # TODO: check that this does not break anything


class Block(nn.Module):
    """ an unassuming Transformer block """
    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.mlp = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            nn.GELU(),  # nice
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(config.resid_pdrop),
        )

    def forward(self, x, layer_past=None, return_present=False):
        # TODO: check that training still works
        if return_present: assert not self.training
        # layer past: tuple of length two with B, nh, T, hs
        attn, present = self.attn(self.ln1(x), layer_past=layer_past)

        x = x + attn
        x = x + self.mlp(self.ln2(x))
        if layer_past is not None or return_present:
            return x, present
        return x


class GPTC(nn.Module):
    """  the continuous GPT model"""
    def __init__(self, vocab_size, block_size, n_layer=12, n_head=8, n_embd=256,
                 embd_pdrop=0., resid_pdrop=0., attn_pdrop=0., n_unmasked=0,
                 n_ind=None, max_seq_len=None, l2_normalized=True,
                 l2_normalize_eps=1e-6) -> None:
        super().__init__()
        if n_ind is None:
            n_ind = n_embd
        if max_seq_len is None:
            max_seq_len = block_size
        # input embedding stem
        # self.input_proj = nn.Linear(config.n_ind, config.n_embd)
        config = GPTCConfig(vocab_size=vocab_size, block_size=block_size,
                           embd_pdrop=embd_pdrop, resid_pdrop=resid_pdrop, attn_pdrop=attn_pdrop,
                           n_layer=n_layer, n_head=n_head, n_embd=n_embd,
                           n_unmasked=n_unmasked, n_ind=n_ind, max_seq_len=max_seq_len,
                           l2_normalized=l2_normalized,
                           l2_normalize_eps=l2_normalize_eps)
        self.input_proj = nn.Identity() if n_ind == n_embd else nn.Linear(n_ind, n_embd)
        self.pos_emb = nn.Parameter(torch.randn(1, config.max_seq_len, config.n_embd) * 0.02) 
        self.drop = nn.Dropout(config.embd_pdrop)
        # transformer
        self.blocks = nn.Sequential(*[Block(config) for _ in range(config.n_layer)])
        # decoder head
        self.ln_f = nn.LayerNorm(config.n_embd)
        
        self.apply(self._init_weights)
        self.config = config
        self.max_seq_length = config.max_seq_len
        self.detach_x = config.detach_x
        self.detach_target = config.detach_target
        self.l2_normalized = config.l2_normalized
        self.l2_normalize_eps = float(config.l2_normalize_eps)
        if self.l2_normalize_eps <= 0:
            raise ValueError("l2_normalize_eps must be positive")

        self.n_classes = config.n_classes
        self.fully_separated = config.fully_separated
        assert not (self.detach_x and self.detach_target), 'Cannot detach both x and target'
        self.head = nn.Linear(config.n_embd, config.n_ind)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, x, targets=None):
        # forward the GPTC model
        # x: (b, n, n_ind)
        if x.shape[1] > self.max_seq_length:
            raise ValueError(
                f"Sequence length {x.shape[1]} exceeds max_seq_len={self.max_seq_length}."
            )
        if x.shape[-1] != self.config.n_ind:
            raise ValueError(f"Expected latent dimension {self.config.n_ind}, got {x.shape[-1]}.")
        token_embeddings = self.input_proj(x)
        # print("token_embeddings shape:", token_embeddings.shape)
        # print("pos_emb shape:", self.pos_emb[:, :token_embeddings.shape[1], :].shape)
        x = self.drop(token_embeddings + self.pos_emb[:, :token_embeddings.shape[1], :])
        x = self.blocks(x)
        x = self.ln_f(x)
        pred = self.head(x) # (b, n, n_ind)
        # if we are given some desired targets also calculate the loss
        loss = None
        if targets is not None:
            loss = F.cross_entropy(pred.view(-1, pred.size(-1)), targets.view(-1))
        return pred, loss
    
    def compute_prior_loss(self, x: torch.Tensor) -> torch.Tensor:
        # x: (b, n, n_ind) 
        if self.l2_normalized:
            x = safe_l2_normalize(x, dim=-1, eps=self.l2_normalize_eps)
            
        target = x[:, 1:]
        if self.detach_target:
            target = target.detach()

        x = x[:, :-1]
        if self.detach_x:
            x = x.detach()

        _, loss = self.forward(x, targets=target)

        return loss

    def ar_predict(self, x: torch.Tensor) -> torch.Tensor:
        # make ar prediction using teacher forcing
        # x: (b, n, n_ind)
        x = x[:, :-1] # (b, n-1, n_ind)
        pred, _ = self.forward(x) # (b, n-1, n_ind)
        full_pred = torch.cat([x[:, :1], pred], dim=1) # (b, n, n_ind)

        if self.l2_normalized:
            full_pred = safe_l2_normalize(
                full_pred, dim=-1, eps=self.l2_normalize_eps
            )
        return full_pred
    


#################################################################################
#                                 GPTC Configs                                  #
#################################################################################   

def GPTC_L(**kwargs):
    return GPTC(GPTCConfig(n_layer=24, n_head=16, n_embd=1024, **kwargs)) # 316.4M?

def GPTC_B(**kwargs):
    return GPTC(GPTCConfig(n_layer=12, n_head=12, n_embd=768, **kwargs)) # 85.9M

def GPTC_M(**kwargs):
    return GPTC(GPTCConfig(n_layer=12, n_head=8, n_embd=512, **kwargs)) # 38.4M

def GPTC_S(**kwargs):
    return GPTC(GPTCConfig(n_layer=12, n_head=6, n_embd=384, **kwargs)) # 21.7M

def GPTC_XS(**kwargs):
    return GPTC(GPTCConfig(n_layer=6, n_head=6, n_embd=384, **kwargs)) # 11.1M

def GPTC_XXS(**kwargs):
    return GPTC(GPTCConfig(n_layer=6, n_head=4, n_embd=256, **kwargs)) # 5.0M
