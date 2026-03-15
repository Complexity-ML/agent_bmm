# Copyright 2026 INL Dynamics / Complexity-ML. All rights reserved.
# Licensed under CC-BY-NC-4.0
"""
Quantized BMM Routing — INT8/INT4 expert dispatch.

Quantizes expert weights to reduce memory and increase throughput.
Uses torch dynamic quantization or manual INT8 matmul.

Memory savings:
  FP32 → INT8: 4x reduction
  FP32 → INT4: 8x reduction (packed)
"""

from __future__ import annotations

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class QuantizedBMMRouter(nn.Module):
    """BMM Router with quantized expert weights."""

    def __init__(
        self,
        hidden_size: int,
        num_tools: int = 4,
        expert_size: int = 256,
        routing: str = "learned",
        quantization: str = "int8",  # "int8", "int4", "none"
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_tools = num_tools
        self.expert_size = expert_size
        self.routing = routing
        self.quantization = quantization

        # Full-precision routing head (small, keep FP32)
        if routing == "learned":
            self.router_head = nn.Linear(hidden_size, num_tools, bias=False)
        elif routing == "embedding":
            self.tool_embeddings = nn.Parameter(torch.empty(num_tools, hidden_size))

        self.gate = nn.Linear(hidden_size, 1, bias=False)

        # Expert weights — will be quantized after init
        self._up_fp32 = nn.Parameter(torch.empty(num_tools, hidden_size, expert_size))
        self._down_fp32 = nn.Parameter(torch.empty(num_tools, expert_size, hidden_size))

        nn.init.kaiming_uniform_(self._up_fp32)
        nn.init.kaiming_uniform_(self._down_fp32)
        nn.init.zeros_(self.gate.weight)

        # Quantized buffers (populated by quantize())
        self.register_buffer("up_q", None)
        self.register_buffer("down_q", None)
        self.register_buffer("up_scale", None)
        self.register_buffer("down_scale", None)

    def quantize(self):
        """Quantize expert weights to INT8 or INT4."""
        if self.quantization == "int8":
            self.up_q, self.up_scale = self._quantize_int8(self._up_fp32.data)
            self.down_q, self.down_scale = self._quantize_int8(self._down_fp32.data)
            logger.info(
                "Quantized to INT8: %.1f MB → %.1f MB",
                self._fp32_size_mb(),
                self._quantized_size_mb(),
            )
        elif self.quantization == "int4":
            self.up_q, self.up_scale = self._quantize_int4(self._up_fp32.data)
            self.down_q, self.down_scale = self._quantize_int4(self._down_fp32.data)
            logger.info(
                "Quantized to INT4: %.1f MB → %.1f MB",
                self._fp32_size_mb(),
                self._quantized_size_mb(),
            )

    @staticmethod
    def _quantize_int8(tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Per-tensor INT8 quantization."""
        scale = tensor.abs().amax() / 127.0
        quantized = (tensor / scale).clamp(-128, 127).to(torch.int8)
        return quantized, scale.unsqueeze(0)

    @staticmethod
    def _quantize_int4(tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Per-tensor INT4 quantization (packed into INT8)."""
        scale = tensor.abs().amax() / 7.0
        quantized = (tensor / scale).clamp(-8, 7).to(torch.int8)
        # Pack two INT4 values into one INT8
        flat = quantized.reshape(-1)
        if flat.shape[0] % 2 != 0:
            flat = F.pad(flat, (0, 1))
        high = (flat[0::2] & 0x0F) << 4
        low = flat[1::2] & 0x0F
        packed = (high | low).to(torch.int8)
        return packed, scale.unsqueeze(0)

    def _dequantize(self, q: torch.Tensor, scale: torch.Tensor, shape: tuple) -> torch.Tensor:
        """Dequantize back to FP32 for computation."""
        if self.quantization == "int8":
            return q.float() * scale
        elif self.quantization == "int4":
            # Unpack INT4
            high = (q >> 4).to(torch.int8)
            low = ((q << 4) >> 4).to(torch.int8)  # sign-extend
            flat = torch.stack([high, low], dim=-1).reshape(-1)
            return flat[: torch.tensor(shape).prod().item()].reshape(shape).float() * scale
        return q.float()

    def route(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Route queries to experts."""
        n = x.shape[0]
        if self.routing == "learned":
            logits = self.router_head(x)
        elif self.routing == "embedding":
            x_norm = F.normalize(x, dim=-1)
            t_norm = F.normalize(self.tool_embeddings, dim=-1)
            logits = x_norm @ t_norm.T
        else:
            ids = torch.arange(n, device=x.device) % self.num_tools
            return ids, torch.ones(n, device=x.device)

        return logits.argmax(dim=-1), torch.ones(n, device=x.device)

    def dispatch(self, x: torch.Tensor, expert_ids: torch.Tensor) -> torch.Tensor:
        """Dispatch with quantized or FP32 weights."""
        if self.up_q is not None:
            up = self._dequantize(self.up_q, self.up_scale, self._up_fp32.shape)
            down = self._dequantize(self.down_q, self.down_scale, self._down_fp32.shape)
        else:
            up = self._up_fp32
            down = self._down_fp32

        sel_up = up[expert_ids]
        sel_down = down[expert_ids]
        h = torch.bmm(x.unsqueeze(1), sel_up).squeeze(1)
        activated = F.silu(h)
        return torch.bmm(activated.unsqueeze(1), sel_down).squeeze(1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        expert_ids, weights = self.route(x)
        expert_out = self.dispatch(x, expert_ids)
        gate_value = torch.sigmoid(self.gate(x))
        output = x + gate_value * expert_out
        return output, expert_ids

    def _fp32_size_mb(self) -> float:
        return (self._up_fp32.numel() + self._down_fp32.numel()) * 4 / 1024 / 1024

    def _quantized_size_mb(self) -> float:
        if self.up_q is None:
            return self._fp32_size_mb()
        return (self.up_q.numel() + self.down_q.numel()) / 1024 / 1024
