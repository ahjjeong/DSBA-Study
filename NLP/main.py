import os
os.environ["TORCH_COMPILE_DISABLE"] = "1"
os.environ["TORCHDYNAMO_DISABLE"] = "1"
from typing import Tuple

import hydra
from omegaconf import DictConfig, OmegaConf

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm
import wandb

from src.utils import seed_everything, get_device, set_wandb
from src.model import EncoderForClassification
from src.data import get_dataloader

def calculate_accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    preds = logits.argmax(dim=-1)
    correct = (preds == labels).sum().item()
    return correct / labels.size(0)


@torch.no_grad()
def run_eval(model: nn.Module, loader, device: torch.device) -> Tuple[float, float]:
    model.eval()
    loss_sum, acc_sum, n = 0.0, 0.0, 0

    for batch in tqdm(loader, desc="Valid/Test", leave=False):
        batch = {k: v.to(device) for k, v in batch.items()}

        logits, loss = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            token_type_ids=batch.get("token_type_ids", None),
            label=batch["labels"],
        )

        bsz = batch["labels"].size(0)
        loss_sum += loss.item() * bsz
        acc_sum += calculate_accuracy(logits, batch["labels"]) * bsz
        n += bsz

    return loss_sum / n, acc_sum / n


def save_checkpoint(path: str, model: nn.Module, optimizer, scheduler, epoch: int, best_val_acc: float):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "epoch": epoch,
            "best_val_acc": best_val_acc,
        },
        path,
    )


def load_checkpoint(path: str, model: nn.Module):
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"])
    return ckpt


@hydra.main(version_base=None, config_path="configs", config_name="default")
def main(cfg: DictConfig):
    # -----------------------------
    # Setup
    # -----------------------------
    seed_everything(int(cfg.seed))
    device = get_device(cfg.device)

    # epochs cap (assignment rule)
    epochs = int(cfg.train.epochs)
    if epochs > 5:
        print(f"[WARN] epochs={epochs} > 5. Clipping to 5 due to assignment rule.")
        epochs = 5

    # -----------------------------
    # Data config merge (dataset + name + seed)
    # data.py needs name for tokenizer
    # -----------------------------
    data_cfg = OmegaConf.to_container(cfg.dataset, resolve=True)  # dict로 풀기
    data_cfg["name"] = cfg.model.name                      # tokenizer/model name
    data_cfg["seed"] = int(cfg.seed)                             # split seed
    data_cfg = OmegaConf.create(data_cfg)                         # 다시 DictConfig로


    train_loader = get_dataloader(cfg.dataset, "train", name=cfg.model.name, seed=cfg.seed)
    valid_loader = get_dataloader(cfg.dataset, "valid", name=cfg.model.name, seed=cfg.seed)
    test_loader  = get_dataloader(cfg.dataset, "test",  name=cfg.model.name, seed=cfg.seed)

    # -----------------------------
    # Model
    # -----------------------------
    model = EncoderForClassification(cfg.model).to(device)

    # -----------------------------
    # Optimizer + constant scheduler
    # -----------------------------
    optimizer = Adam(
        model.parameters(),
        lr=float(cfg.optimizer.lr),
        betas=tuple(cfg.optimizer.betas),
        eps=float(cfg.optimizer.eps),
        weight_decay=float(cfg.optimizer.weight_decay),
    )
    scheduler = LambdaLR(optimizer, lr_lambda=lambda step: 1.0)

    # -----------------------------
    # W&B
    # -----------------------------
    use_wandb = set_wandb(model, cfg)

    # -----------------------------
    # Checkpoint paths
    # hydra.run.dir: . 로 설정해놨으니 상대경로 그대로 프로젝트 폴더에 생김
    # -----------------------------
    ckpt_dir = os.path.join("checkpoints", cfg.model.name.replace("/", "_"))
    best_ckpt_path = os.path.join(ckpt_dir, "best.pt")

    # -----------------------------
    # Training loop
    # -----------------------------
    best_val_acc = -1.0
    global_step = 0

    for epoch in range(1, epochs + 1):
        model.train()
        loss_sum, acc_sum, n = 0.0, 0.0, 0

        pbar = tqdm(train_loader, desc=f"Train Epoch {epoch}/{epochs}")
        for batch in pbar:
            batch = {k: v.to(device) for k, v in batch.items()}

            logits, loss = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                token_type_ids=batch.get("token_type_ids", None),
                label=batch["labels"],
            )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            scheduler.step()

            bsz = batch["labels"].size(0)
            acc = calculate_accuracy(logits, batch["labels"])

            loss_sum += loss.item() * bsz
            acc_sum += acc * bsz
            n += bsz
            global_step += 1

            pbar.set_postfix({"loss": f"{loss.item():.4f}", "acc": f"{acc:.4f}"})

            if use_wandb:
                wandb.log(
                    {
                        "train/loss_step": loss.item(),
                        "train/acc_step": acc,
                        "lr": optimizer.param_groups[0]["lr"],
                        "epoch": epoch,
                        "step": global_step,
                    },
                    step=global_step,
                )

        train_loss = loss_sum / n
        train_acc = acc_sum / n

        # -----------------------------
        # Validation
        # -----------------------------
        val_loss, val_acc = run_eval(model, valid_loader, device)

        print(
            f"[Epoch {epoch}] "
            f"train_loss={train_loss:.4f}, train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f}, val_acc={val_acc:.4f}"
        )

        if use_wandb:
            wandb.log(
                {
                    "train/loss_epoch": train_loss,
                    "train/acc_epoch": train_acc,
                    "val/loss": val_loss,
                    "val/acc": val_acc,
                    "epoch": epoch,
                },
                step=global_step,
            )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_checkpoint(best_ckpt_path, model, optimizer, scheduler, epoch, best_val_acc)
            print(f"✅ Best checkpoint updated: val_acc={best_val_acc:.4f} @ epoch={epoch}")

    # -----------------------------
    # Test with best checkpoint
    # -----------------------------
    print(f"\nLoading best checkpoint: {best_ckpt_path}")
    load_checkpoint(best_ckpt_path, model)
    test_loss, test_acc = run_eval(model, test_loader, device)

    print(f"[Test] loss={test_loss:.4f}, acc={test_acc:.4f}")

    if use_wandb:
        wandb.log({"test/loss": test_loss, "test/acc": test_acc})
        wandb.finish()


if __name__ == "__main__":
    main()