import torch


@torch.no_grad()
def accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    preds = logits.argmax(dim=1)
    return (preds == targets).float().mean().item()


@torch.no_grad()
def top_k_error(
    logits: torch.Tensor,
    targets: torch.Tensor,
    k: int = 5,
) -> float:
    """
    Top-k error = 1 - Top-k accuracy
    """
    if logits.ndim != 2:
        raise ValueError(f"logits must be 2D (B, C), got {logits.shape}")
    if targets.ndim != 1:
        raise ValueError(f"targets must be 1D (B,), got {targets.shape}")

    B, C = logits.shape
    kk = min(int(k), int(C))  # class 수보다 k가 커지지 않도록

    topk = logits.topk(kk, dim=1).indices
    correct = topk.eq(targets.view(-1, 1)).any(dim=1)
    topk_acc = correct.float().mean().item()

    return 1.0 - topk_acc