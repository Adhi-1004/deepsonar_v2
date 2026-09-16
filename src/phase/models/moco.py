from __future__ import annotations

import copy

import torch
import torch.nn as nn

from phase.config import MocoConfig
from phase.models.projection import ProjectionHead
from phase.models.resnet import EMBEDDING_DIM, build_backbone


def split_batchnorm(module: nn.Module, splits: int = 8) -> nn.Module:
    """Replace BatchNorm with a split variant.

    MoCo shuffles batch-norm statistics across GPUs so the query encoder cannot
    cheat by reading the batch composition. On a single GPU that trick is
    unavailable, and without a substitute the model leaks through BN statistics
    and the contrastive task collapses to a shortcut.
    """
    for name, child in module.named_children():
        if isinstance(child, nn.BatchNorm2d):
            setattr(module, name, SplitBatchNorm2d(child.num_features, splits))
        else:
            split_batchnorm(child, splits)
    return module


class SplitBatchNorm2d(nn.BatchNorm2d):
    def __init__(self, num_features: int, splits: int):
        super().__init__(num_features)
        self.splits = splits

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or x.shape[0] % self.splits:
            return super().forward(x)

        n, c, h, w = x.shape
        chunk = n // self.splits
        running_mean = self.running_mean.repeat(self.splits)
        running_var = self.running_var.repeat(self.splits)

        out = nn.functional.batch_norm(
            x.view(1, self.splits * c, chunk * h, w),
            running_mean,
            running_var,
            self.weight.repeat(self.splits),
            self.bias.repeat(self.splits),
            True,
            self.momentum,
            self.eps,
        )
        self.running_mean.copy_(running_mean.view(self.splits, c).mean(dim=0))
        self.running_var.copy_(running_var.view(self.splits, c).mean(dim=0))
        return out.view(n, c, h, w)


class MoCo(nn.Module):
    def __init__(self, config: MocoConfig, in_channels: int = 1):
        super().__init__()
        self.config = config
        self.momentum = config.momentum
        self.temperature = config.temperature

        embedding = EMBEDDING_DIM[config.backbone]
        encoder = build_backbone(config.backbone, in_channels)
        if config.norm == "splitbn":
            encoder = split_batchnorm(encoder)

        self.encoder_q = nn.Sequential(
            encoder, ProjectionHead(embedding, config.mlp_hidden, config.dim)
        )
        self.encoder_k = copy.deepcopy(self.encoder_q)
        for param in self.encoder_k.parameters():
            param.requires_grad = False

        self.register_buffer(
            "queue", nn.functional.normalize(torch.randn(config.dim, config.queue_size), dim=0)
        )
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def update_key_encoder(self) -> None:
        for query, key in zip(
            self.encoder_q.parameters(), self.encoder_k.parameters(), strict=True
        ):
            key.data.mul_(self.momentum).add_(query.data, alpha=1.0 - self.momentum)
        for query, key in zip(self.encoder_q.buffers(), self.encoder_k.buffers(), strict=True):
            key.data.copy_(query.data)

    @torch.no_grad()
    def enqueue(self, keys: torch.Tensor) -> None:
        batch = keys.shape[0]
        size = self.queue.shape[1]
        pointer = int(self.queue_ptr)

        end = pointer + batch
        if end <= size:
            self.queue[:, pointer:end] = keys.T
        else:
            first = size - pointer
            self.queue[:, pointer:] = keys[:first].T
            self.queue[:, : end - size] = keys[first:].T
        self.queue_ptr[0] = end % size

    def embed(self, features: torch.Tensor) -> torch.Tensor:
        return self.encoder_q[0](features)

    def forward(self, view_q: torch.Tensor, view_k: torch.Tensor):
        query = self.encoder_q(view_q)

        with torch.no_grad():
            self.update_key_encoder()
            shuffle = torch.randperm(view_k.shape[0], device=view_k.device)
            unshuffle = torch.argsort(shuffle)
            key = self.encoder_k(view_k[shuffle])[unshuffle]

        positive = torch.einsum("nc,nc->n", query, key).unsqueeze(-1)
        negative = torch.einsum("nc,ck->nk", query, self.queue.clone().detach())
        logits = torch.cat([positive, negative], dim=1) / self.temperature
        labels = torch.zeros(logits.shape[0], dtype=torch.long, device=logits.device)
        return logits, labels, query, key
