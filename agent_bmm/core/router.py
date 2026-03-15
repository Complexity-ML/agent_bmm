# Copyright 2026 INL Dynamics / Complexity-ML. All rights reserved.
# Licensed under CC-BY-NC-4.0
"""
BMM Router — Batched Matrix Multiply tool dispatch.

Routes queries to tool experts via deterministic or learned routing,
then dispatches ALL queries in parallel via BMM. Zero sequential loops.

Routing strategies:
    - learned:     Router MLP picks the best tool per query
    - round_robin: Deterministic position-based (like I64)
    - embedding:   Cosine similarity between query and tool descriptions
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class BMMRouter(nn.Module):
    """
    Parallel tool dispatch via Batched Matrix Multiply.

    Each tool is a small expert MLP. The router selects which expert
    processes each query, then BMM executes ALL queries in parallel.

    Args:
        hidden_size: Embedding dimension of input queries.
        num_tools: Number of tool experts.
        expert_size: Internal dimension of each tool expert.
        routing: "learned", "round_robin", or "embedding".
        top_k: Number of tools to activate per query (1 = hard routing).
    """

    def __init__(
        self,
        hidden_size: int,
        num_tools: int = 4,
        expert_size: int = 256,
        routing: str = "learned",
        top_k: int = 1,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_tools = num_tools
        self.expert_size = expert_size
        self.routing = routing
        self.top_k = top_k

        # Expert weights packed for BMM
        self.up_proj = nn.Parameter(torch.empty(num_tools, hidden_size, expert_size))
        self.down_proj = nn.Parameter(torch.empty(num_tools, expert_size, hidden_size))

        # Routing head
        if routing == "learned":
            self.router_head = nn.Linear(hidden_size, num_tools, bias=False)
        elif routing == "embedding":
            self.tool_embeddings = nn.Parameter(torch.empty(num_tools, hidden_size))

        # Gated residual — starts at 0 (safe plug-in, no disruption)
        self.gate = nn.Linear(hidden_size, 1, bias=False)

        self._init_weights()

    def _init_weights(self):
        nn.init.kaiming_uniform_(self.up_proj)
        nn.init.kaiming_uniform_(self.down_proj)
        nn.init.zeros_(self.gate.weight)
        if hasattr(self, "router_head"):
            nn.init.normal_(self.router_head.weight, std=0.01)
        if hasattr(self, "tool_embeddings"):
            nn.init.normal_(self.tool_embeddings, std=0.02)

    def route(
        self,
        x: torch.Tensor,
        positions: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute routing decisions.

        Returns:
            expert_ids: (N,) or (N, top_k) — selected tool indices
            expert_weights: (N,) or (N, top_k) — routing weights (for top_k > 1)
        """
        N = x.shape[0]

        if self.routing == "learned":
            logits = self.router_head(x)  # (N, num_tools)
        elif self.routing == "embedding":
            # Cosine similarity between query and tool descriptions
            x_norm = F.normalize(x, dim=-1)
            t_norm = F.normalize(self.tool_embeddings, dim=-1)
            logits = x_norm @ t_norm.T  # (N, num_tools)
        else:
            # Round-robin deterministic routing
            if positions is not None:
                ids = (positions % self.num_tools).long()
            else:
                ids = torch.arange(N, device=x.device) % self.num_tools
            weights = torch.ones(N, device=x.device)
            return ids, weights

        if self.top_k == 1:
            expert_ids = logits.argmax(dim=-1)  # (N,)
            expert_weights = torch.ones(N, device=x.device)
        else:
            expert_weights, expert_ids = logits.topk(self.top_k, dim=-1)
            expert_weights = F.softmax(expert_weights, dim=-1)

        return expert_ids, expert_weights

    def dispatch(
        self,
        x: torch.Tensor,
        expert_ids: torch.Tensor,
    ) -> torch.Tensor:
        """
        BMM dispatch — all queries processed in parallel.

        Args:
            x: (N, H) input queries
            expert_ids: (N,) selected tool per query

        Returns:
            (N, H) expert outputs
        """
        # Select each query's expert weights
        sel_up = self.up_proj[expert_ids]  # (N, H, E)
        sel_down = self.down_proj[expert_ids]  # (N, E, H)

        # Up: (N, 1, H) @ (N, H, E) → (N, E)
        up = torch.bmm(x.unsqueeze(1), sel_up).squeeze(1)
        activated = F.silu(up)

        # Down: (N, 1, E) @ (N, E, H) → (N, H)
        return torch.bmm(activated.unsqueeze(1), sel_down).squeeze(1)

    def forward(
        self,
        x: torch.Tensor,
        positions: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Full routing + dispatch + gated residual.

        Returns:
            output: (N, H) — input augmented with tool expert output
            expert_ids: (N,) — which tool was selected per query
        """
        expert_ids, expert_weights = self.route(x, positions)

        if self.top_k == 1:
            expert_out = self.dispatch(x, expert_ids)
        else:
            # Multi-tool: weighted sum of top_k experts
            expert_out = torch.zeros_like(x)
            for k in range(self.top_k):
                ids_k = expert_ids[:, k]
                w_k = expert_weights[:, k].unsqueeze(-1)  # (N, 1)
                expert_out = expert_out + w_k * self.dispatch(x, ids_k)

        gate_value = torch.sigmoid(self.gate(x))  # (N, 1)
        output = x + gate_value * expert_out
        return output, expert_ids
