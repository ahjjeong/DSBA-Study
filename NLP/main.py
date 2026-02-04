import os
import torch
import hydra
import wandb
from tqdm import tqdm
import omegaconf
from omegaconf import DictConfig, OmegaConf

# torch.cuda.set_per_process_memory_fraction(11/24) -> 김재희 로컬과 신입생 로컬의 vram 맞추기 용도. 과제 수행 시 삭제하셔도 됩니다. 
# model과 data에서 정의된 custom class 및 function을 import합니다.

from src.model import EncoderForClassification
from src.data import get_dataloader
from src.utils import seed_everything, get_device, set_wandb


def train_iter(model, batch, optimizer, device, global_step: int):
    model.train()
    batch = {k: v.to(device) for k, v in batch.items()}
    logits, loss = model(**batch)  # tuple 반환 가정
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    # train acc (iter 기준)
    acc = calculate_accuracy(logits, batch["labels"])
    # wandb: iter(step) 단위로 loss/acc 기록
    wandb.log(
        {
            "train/loss_step": loss.item(),
            "train/acc_step": acc,
        },
        step=global_step,
    )
    return loss.item(), acc

@torch.no_grad()
def valid_iter(model, batch, device):
    model.eval()
    batch = {k: v.to(device) for k, v in batch.items()}
    logits, loss = model(**batch)
    acc = calculate_accuracy(logits, batch["labels"])
    return loss.item(), acc

def calculate_accuracy(logits, label):
    preds = logits.argmax(dim=-1)
    correct = (preds == label).sum().item()
    return correct / label.size(0)

@hydra.main(version_base=None, config_path="configs", config_name="default")
def main(configs: omegaconf.DictConfig):
    print(OmegaConf.to_yaml(configs))

    # basic setup
    seed_everything(int(getattr(configs, "seed", 42)))
    device = get_device(configs.device)

    # model
    num_labels = int(getattr(configs.dataset, "num_labels", 2))
    model = EncoderForClassification(configs.model, num_labels=num_labels).to(device)

    # data
    train_loader = get_dataloader(configs, split="train", model_name=configs.model.name)
    valid_loader = get_dataloader(configs, split="valid", model_name=configs.model.name)
    test_loader = get_dataloader(configs, split="test", model_name=configs.model.name)

    # optimizer
    opt_name = str(configs.optimizer.name).lower()
    if opt_name == "adam":
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=float(configs.optimizer.lr),
            betas=tuple(configs.optimizer.betas),
            eps=float(configs.optimizer.eps),
            weight_decay=float(configs.optimizer.weight_decay),
        )
    else:
        raise ValueError(f"Unsupported optimizer: {configs.optimizer.name}")

    # wandb
    use_wandb = set_wandb(configs)

    # training loop
    best_val_acc = -1.0
    global_step = 0

    for epoch in range(int(configs.train.epochs)):
        # train
        train_loss_sum, train_acc_sum, n_train = 0.0, 0.0, 0

        for batch in tqdm(train_loader, desc=f"Train [Epoch {epoch+1}]"):
            if not use_wandb:
                # wandb 안 쓰면 log 호출 안 하도록(에러 방지)
                batch_device = {k: v.to(device) for k, v in batch.items()}
                logits, loss = model(**batch_device)
                acc = calculate_accuracy(logits, batch_device["labels"])
                loss_val, acc_val = loss.item(), acc
            else:
                loss_val, acc_val = train_iter(model, batch, optimizer, device, global_step)

            bsz = batch["labels"].size(0)
            train_loss_sum += loss_val * bsz
            train_acc_sum += acc_val * bsz
            n_train += bsz
            global_step += 1

        train_loss = train_loss_sum / n_train
        train_acc = train_acc_sum / n_train

        # valid
        val_loss_sum, val_acc_sum, n_val = 0.0, 0.0, 0

        for batch in tqdm(valid_loader, desc=f"Valid [Epoch {epoch+1}]"):
            loss_val, acc_val = valid_iter(model, batch, device)

            bsz = batch["labels"].size(0)
            val_loss_sum += loss_val * bsz
            val_acc_sum += acc_val * bsz
            n_val += bsz

        val_loss = val_loss_sum / n_val
        val_acc = val_acc_sum / n_val

        print(
            f"[Epoch {epoch+1}] "
            f"train_loss={train_loss:.4f}, train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f}, val_acc={val_acc:.4f}"
        )

        # wandb: epoch 단위로도 기록
        if use_wandb:
            wandb.log(
                {
                    "epoch": epoch + 1,
                    "train/loss_epoch": train_loss,
                    "train/acc_epoch": train_acc,
                    "val/loss_epoch": val_loss,
                    "val/acc_epoch": val_acc,
                },
                step=global_step,
            )

        # best checkpoint
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), os.path.join(os.getcwd(), "best.pt"))

    # final test
    test_loss_sum, test_acc_sum, n_test = 0.0, 0.0, 0

    for batch in tqdm(test_loader, desc="Test"):
        loss_val, acc_val = valid_iter(model, batch, device)

        bsz = batch["labels"].size(0)
        test_loss_sum += loss_val * bsz
        test_acc_sum += acc_val * bsz
        n_test += bsz

    test_loss = test_loss_sum / n_test
    test_acc = test_acc_sum / n_test

    print(f"[Test] loss={test_loss:.4f}, acc={test_acc:.4f}")

    if use_wandb:
        wandb.log(
            {
                "test/loss": test_loss,
                "test/acc": test_acc,
            },
            step=global_step,
        )
    
if __name__ == "__main__" :
    main()