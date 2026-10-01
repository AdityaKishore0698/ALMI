import math
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.distributions import Categorical
import models.pos_encoding as pos_encoding

class ALMITransformer(nn.Module):
    def __init__(self, 
                num_obs=1024, 
                embed_dim=512, 
                clip_dim=512, 
                block_size=16,  # max motion seq
                num_layers=2, 
                n_head=8, 
                drop_out_rate=0.1, 
                fc_rate=4,
                pred_action=False,
                chunk=1):
        super().__init__()
        self.trans_base = CrossCondTransBase(num_obs, embed_dim, clip_dim, block_size, num_layers, n_head, drop_out_rate, fc_rate)
        if pred_action:
            # action chunking: each position predicts the next `chunk` actions (21 each); chunk=1 is the original model
            self.trans_head = CrossCondTransHead((num_obs-50) * chunk, embed_dim, block_size, num_layers, n_head, drop_out_rate, fc_rate)
        else:
            self.trans_head = CrossCondTransHead(num_obs, embed_dim, block_size, num_layers, n_head, drop_out_rate, fc_rate)
        self.block_size = block_size
        self.num_obs = num_obs
        self.chunk = chunk

    def set_text_norm(self, mean, std):
        """Standardise CLIP text features with statistics of the training captions (see CrossCondTransBase)."""
        self.trans_base.set_text_norm(mean, std)

    def get_block_size(self):
        return self.block_size

    def forward(self, idxs, clip_feature):
        feat = self.trans_base(idxs, clip_feature)
        out = self.trans_head(feat)
        return out

    def sample(self, clip_feature, if_categorial=False):
        for k in range(self.block_size):
            if k == 0:
                x = []
            else:
                x = xs
            logits = self.forward(x, clip_feature)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            if if_categorial:
                dist = Categorical(probs)
                idx = dist.sample()
                if idx == self.num_vq:
                    break
                idx = idx.unsqueeze(-1)
            else:
                _, idx = torch.topk(probs, k=1, dim=-1)
                if idx[0] == self.num_vq:
                    break
            # append to the sequence and continue
            if k == 0:
                xs = idx
            else:
                xs = torch.cat((xs, idx), dim=1)
            
            if k == self.block_size - 1:
                return xs[:, :-1]
        return xs

class CausalConditionalSelfAttention(nn.Module):

    def __init__(self, embed_dim=512, block_size=16, n_head=8, drop_out_rate=0.1):
        super().__init__()
        assert embed_dim % 8 == 0
        # key, query, value projections for all heads
        self.key = nn.Linear(embed_dim, embed_dim)
        self.query = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)

        self.attn_drop = nn.Dropout(drop_out_rate)
        self.resid_drop = nn.Dropout(drop_out_rate)

        self.proj = nn.Linear(embed_dim, embed_dim)
        # causal mask to ensure that attention is only applied to the left in the input sequence
        self.register_buffer("mask", torch.tril(torch.ones(block_size, block_size)).view(1, 1, block_size, block_size))
        self.n_head = n_head

    def forward(self, x):
        B, T, C = x.size() 

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        k = self.key(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.mask[:,:,:T,:T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_drop(self.proj(y))
        return y

class Block(nn.Module):

    def __init__(self, embed_dim=512, block_size=16, n_head=8, drop_out_rate=0.1, fc_rate=4):
        super().__init__()
        self.ln1 = nn.LayerNorm(embed_dim)
        self.ln2 = nn.LayerNorm(embed_dim)
        self.attn = CausalConditionalSelfAttention(embed_dim, block_size, n_head, drop_out_rate)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, fc_rate * embed_dim),
            nn.GELU(),
            nn.Linear(fc_rate * embed_dim, embed_dim),
            nn.Dropout(drop_out_rate),
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x

class CrossCondTransBase(nn.Module):

    def __init__(self, 
                num_obs=1024, 
                embed_dim=512, 
                clip_dim=512, 
                block_size=16, 
                num_layers=2, 
                n_head=8, 
                drop_out_rate=0.1, 
                fc_rate=4):
        super().__init__()
        # self.tok_emb = nn.Embedding(num_vq + 2, embed_dim)
        self.tok_emb = nn.Linear(num_obs, embed_dim)
        self.cond_emb = nn.Linear(clip_dim, embed_dim)
        # self.pos_embedding = nn.Embedding(block_size, embed_dim)
        self.drop = nn.Dropout(drop_out_rate)
        # transformer block
        self.blocks = nn.Sequential(*[Block(embed_dim, block_size, n_head, drop_out_rate, fc_rate) for _ in range(num_layers)])
        self.pos_embed = pos_encoding.PositionEmbedding(block_size, embed_dim, 0.0, False)
        # CLIP features of the training captions are nearly identical (e.g. cosine 0.996 between "wave left" and
        # "wave right"), so the hand instruction barely reaches the model. Standardising each feature dimension with
        # the captions' mean and std makes those differences as large as the rest. Identity unless set_text_norm()
        # is called; the statistics are then saved in the checkpoint (old checkpoints load unchanged).
        self.register_buffer("text_mean", torch.zeros(clip_dim), persistent=False)
        self.register_buffer("text_std", torch.ones(clip_dim), persistent=False)

        self.block_size = block_size

        self.apply(self._init_weights)

    def set_text_norm(self, mean, std):
        self.register_buffer("text_mean", mean.detach().clone().float(), persistent=True)
        self.register_buffer("text_std", std.detach().clone().float(), persistent=True)

    def get_block_size(self):
        return self.block_size

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
    
    def forward(self, idx, clip_feature):
        clip_feature = (clip_feature - self.text_mean) / self.text_std
        if len(idx) == 0:
            token_embeddings = self.cond_emb(clip_feature).unsqueeze(1)
        else:
            # b, t, _ = idx.size()
            t = idx.shape[1]
            assert t <= self.block_size, "Cannot forward, model block size is exhausted."
            # forward the Trans model
            token_embeddings = self.tok_emb(idx)
            token_embeddings = torch.cat([self.cond_emb(clip_feature).unsqueeze(1), token_embeddings], dim=1)
            
        x = self.pos_embed(token_embeddings)
        x = self.blocks(x)

        return x

class CrossCondTransHead(nn.Module):

    def __init__(self, 
                num_obs=1024, 
                embed_dim=512, 
                block_size=16, 
                num_layers=2, 
                n_head=8, 
                drop_out_rate=0.1, 
                fc_rate=4):
        super().__init__()

        self.blocks = nn.Sequential(*[Block(embed_dim, block_size, n_head, drop_out_rate, fc_rate) for _ in range(num_layers)])
        self.ln_f = nn.LayerNorm(embed_dim)
        # self.head = nn.Linear(embed_dim, num_obs, bias=False)
        self.head = nn.Linear(embed_dim, num_obs)
        self.block_size = block_size

        self.apply(self._init_weights)

    def get_block_size(self):
        return self.block_size

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, x):
        x = self.blocks(x)
        x = self.ln_f(x)
        out = self.head(x)
        return out


def load_trans_state(model, state):
    """load_state_dict that also restores text-feature statistics saved by set_text_norm()."""
    if 'trans_base.text_mean' in state:
        model.set_text_norm(state['trans_base.text_mean'], state['trans_base.text_std'])
    model.load_state_dict(state, strict=True)


def chunk_from_state(state, num_actions=21):
    """Chunk size of a checkpoint, from the width of its output layer."""
    return state['trans_head.head.weight'].shape[0] // num_actions
