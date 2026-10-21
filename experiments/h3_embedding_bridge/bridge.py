"""Experimental 9B hidden-state -> H3 layer-50 bridge, not an encoder replacement yet."""
import math
from dataclasses import asdict, dataclass
import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class BridgeConfig:
    student_dim: int = 4096
    teacher_dim: int = 5120
    width: int = 512
    heads: int = 8
    source_slots: int = 512
    query_chunk: int = 256


class HiddenStateBridge(nn.Module):
    """Queries use only target positions/modality tags, NEVER teacher features.

    Teacher layout is needed for this cached-case prototype. Arbitrary-input
    layout prediction is deliberately not claimed or implemented.
    """
    def __init__(self, config=BridgeConfig()):
        super().__init__()
        self.config = config
        self.source = nn.Sequential(nn.LayerNorm(config.student_dim), nn.Linear(config.student_dim, config.width))
        self.modality = nn.Embedding(2, config.width)
        self.attention = nn.MultiheadAttention(config.width, config.heads, batch_first=True)
        self.norm = nn.LayerNorm(config.width)
        self.output = nn.Linear(config.width, config.teacher_dim)
        self.register_buffer('teacher_channel_scale', torch.ones(config.teacher_dim))

    def position(self, length, device):
        # Normalized positions do not depend on teacher feature values.
        pos = torch.linspace(0, 1, length, device=device)[:, None]
        freq = torch.exp(torch.arange(0, self.config.width, 2, device=device) * (-math.log(10000) / self.config.width))
        angles = pos * 1000 * freq
        return torch.stack((angles.sin(), angles.cos()), -1).flatten(-2)

    def memory(self, prefix, generated):
        # Separate quotas keep thinking states from being washed out by image tokens.
        banks = [self.source(prefix)]
        if generated.shape[0]:
            banks.append(self.source(generated))
        quota = max(1, self.config.source_slots // len(banks))
        pooled = [F.adaptive_avg_pool1d(x.T[None], min(quota, len(x))).transpose(1, 2) for x in banks]
        result = torch.cat(pooled, 1)
        return result + self.position(result.shape[1], result.device)[None].to(result.dtype)

    def forward(self, prefix, generated, target_tags):
        assert prefix.ndim == generated.ndim == 2
        assert prefix.shape[-1] == generated.shape[-1] == self.config.student_dim
        assert target_tags.ndim == 1 and len(target_tags) > 0
        memory = self.memory(prefix, generated)
        queries = self.modality(target_tags.long()) + self.position(len(target_tags), target_tags.device).to(memory.dtype)
        output = []
        for start in range(0, len(queries), self.config.query_chunk):
            q = queries[None, start:start + self.config.query_chunk]
            x, _ = self.attention(q, memory, memory, need_weights=False)
            output.append(self.output(self.norm(x + q))[0])
        return torch.cat(output) * self.teacher_channel_scale

    def config_dict(self):
        return asdict(self.config)


def alignment_loss(prediction, teacher, tags):
    # Equal modality weighting prevents the longer visual prefix dominating text.
    scale = teacher.float().square().mean(0).sqrt().clamp_min(1e-3)
    losses = []
    for kind in (0, 1):
        mask = tags == kind
        if mask.any():
            p, t = prediction[mask].float(), teacher[mask].float()
            losses.append(((p - t) / scale).square().mean() + .1 * (1 - F.cosine_similarity(p, t).mean()))
    return torch.stack(losses).mean()
