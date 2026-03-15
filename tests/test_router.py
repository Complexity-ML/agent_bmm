# Copyright 2026 INL Dynamics / Complexity-ML. All rights reserved.
import torch
import sys
sys.path.insert(0, ".")

from agent_bmm.core.router import BMMRouter


def test_round_robin():
    router = BMMRouter(hidden_size=256, num_tools=4, routing="round_robin")
    x = torch.randn(8, 256)
    positions = torch.arange(8)
    output, ids = router(x, positions)
    assert ids.tolist() == [0, 1, 2, 3, 0, 1, 2, 3]
    assert output.shape == (8, 256)
    print("OK: round_robin routing")


def test_learned_routing():
    router = BMMRouter(hidden_size=256, num_tools=4, routing="learned")
    x = torch.randn(16, 256)
    output, ids = router(x)
    assert output.shape == (16, 256)
    assert ids.shape == (16,)
    assert all(0 <= i < 4 for i in ids.tolist())
    print(f"OK: learned routing, distribution={[ids.tolist().count(i) for i in range(4)]}")


def test_embedding_routing():
    router = BMMRouter(hidden_size=256, num_tools=4, routing="embedding")
    x = torch.randn(8, 256)
    output, ids = router(x)
    assert output.shape == (8, 256)
    print(f"OK: embedding routing, ids={ids.tolist()}")


def test_top_k():
    router = BMMRouter(hidden_size=256, num_tools=4, routing="learned", top_k=2)
    x = torch.randn(8, 256)
    output, ids = router(x)
    assert output.shape == (8, 256)
    assert ids.shape == (8, 2)
    print(f"OK: top_k=2 routing")


def test_gate_init_zero():
    router = BMMRouter(hidden_size=256, num_tools=4)
    x = torch.randn(8, 256)
    output, _ = router(x)
    diff = (output - x).norm() / x.norm()
    assert diff < 0.5, f"Gate should start near zero, diff={diff:.4f}"
    print(f"OK: gate init near zero, diff={diff:.4f}")


def test_batch_sizes():
    router = BMMRouter(hidden_size=256, num_tools=4)
    for n in [1, 2, 4, 8, 16, 32, 64, 128, 256]:
        x = torch.randn(n, 256)
        output, ids = router(x)
        assert output.shape == (n, 256)
    print("OK: all batch sizes [1..256]")


def test_no_nan():
    router = BMMRouter(hidden_size=768, num_tools=8, expert_size=512)
    x = torch.randn(64, 768)
    output, ids = router(x)
    assert not output.isnan().any(), "NaN in output"
    assert not output.isinf().any(), "Inf in output"
    print("OK: no NaN/Inf")


if __name__ == "__main__":
    test_round_robin()
    test_learned_routing()
    test_embedding_routing()
    test_top_k()
    test_gate_init_zero()
    test_batch_sizes()
    test_no_nan()
    print("\nAll router tests passed!")
