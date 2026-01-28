import math
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim

from src.metrics import accuracy, top_k_error


@torch.no_grad()
def evaluate(model, loader, device, criterion, k=5):
    """
    validation/test 평가 함수
    """
    model.eval()

    # 초기화
    loss_sum = 0.0
    acc_sum = 0.0
    topk_err_sum = 0.0
    total = 0

    for x, y in loader:
        # GPU로 이동
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        logits = model(x)
        loss = criterion(logits, y)

        bsz = x.size(0)
        loss_sum += loss.item() * bsz
        acc_sum += accuracy(logits, y) * bsz  # top-1 acc

        topk_err = top_k_error(logits, y, k)
        topk_err_sum += topk_err * bsz

        total += bsz

    loss_avg = loss_sum / total
    acc_avg = acc_sum / total
    topk_err_avg = topk_err_sum / total

    metrics = {
        "loss": loss_avg,
        "acc": acc_avg,
        "top1_err": 1.0 - acc_avg,
        f"top{k}_err": topk_err_avg,
    }

    return metrics


def build_criterion(cfg_train) -> nn.Module:
    '''
    loss 함수 만들기
    '''
    name = str(cfg_train.loss_function)
    if name == "CrossEntropyLoss":
        return nn.CrossEntropyLoss()
    raise ValueError(f"Unsupported loss_function: {name}")


def build_optimizer(cfg_optimizer, model: nn.Module) -> optim.Optimizer:
    name = str(cfg_optimizer.name).lower()

    if name == "adam":
        return optim.Adam(
            model.parameters(),
            lr=float(cfg_optimizer.lr),
            betas=tuple(cfg_optimizer.betas),
            eps=float(cfg_optimizer.eps),
            weight_decay=float(getattr(cfg_optimizer, "weight_decay", 0.0)),
        )

    raise ValueError(f"Unsupported optimizer: {cfg_optimizer.name}")


def build_warmup_cosine_scheduler(optimizer, epochs, warmup_epochs):
    '''
    epoch 기준으로 lr을 조절하는 규칙 생성
    - warmup 구간: lr 배율이 0 -> 1로 선형 증가
    - 이후 구간: cosine으로 1 -> 0으로 서서히 감소
    '''
    warmup_epochs = max(0, int(warmup_epochs))
    epochs = int(epochs)

    def lr_lambda(epoch: int) -> float:
        if warmup_epochs > 0 and epoch < warmup_epochs:
            return float(epoch + 1) / float(warmup_epochs)
        if epochs <= warmup_epochs:
            return 1.0
        progress = (epoch - warmup_epochs) / float(epochs - warmup_epochs)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


def train(
    model: nn.Module,
    train_loader,
    val_loader,
    cfg_train,
    cfg_opt,
    device: torch.device,
    wandb_run=None,
    k=5,
):
    """
    train 함수
    """
    # criterion, optimizer, scheduler, amp 준비
    criterion = build_criterion(cfg_train)
    optimizer = build_optimizer(cfg_opt, model)
    scheduler = build_warmup_cosine_scheduler(
        optimizer,
        epochs=int(cfg_train.epochs),
        warmup_epochs=int(cfg_train.warmup_epochs),
    )
    scaler = torch.cuda.amp.GradScaler(enabled=bool(cfg_train.amp) and device.type == "cuda")

    # 초기화
    best_val_acc = -1.0
    best_state = None

    for epoch in range(int(cfg_train.epochs)):
        # train
        model.train()
        loss_sum = 0.0
        total = 0

        pbar = tqdm(train_loader, desc=f"train e{epoch+1}", leave=False)
        for x, y in pbar:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            if bool(cfg_train.amp) and device.type == "cuda":
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    logits = model(x)
                    loss = criterion(logits, y)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                logits = model(x)
                loss = criterion(logits, y)
                loss.backward()
                optimizer.step()

            bsz = x.size(0)
            loss_sum += loss.item() * bsz
            total += bsz
            pbar.set_postfix(loss=float(loss.item()))

        train_loss = loss_sum / total

        # validation
        val_metrics = evaluate(model, val_loader, device, criterion, k=k)
        val_loss = val_metrics["loss"]
        val_acc = val_metrics["acc"]

        # epoch 끝날 때 learning rate 업데이트
        scheduler.step()

        # best model 갱신
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {kk: vv.detach().cpu() for kk, vv in model.state_dict().items()}
        
        topk_key = f"top{k}_err"

        # logging
        log_payload = {
            "epoch": epoch + 1,
            "train/loss": train_loss,
            "val/loss": val_loss,
            "val/acc": val_acc,
            "val/top1_err": val_metrics["top1_err"],
            f"val/{topk_key}": val_metrics[topk_key],
            "best_val/acc": best_val_acc,
            "lr": float(optimizer.param_groups[0]["lr"]),
        }

        if wandb_run is not None:
            wandb_run.log(log_payload)

        # print
        print(
            f"[epoch {epoch+1:02d}] "
            f"train_loss={train_loss:.4f} "
            f"val_loss={val_loss:.4f} "
            f"val_acc={val_acc:.4f} "
            f"top1_err={val_metrics['top1_err']:.4f} "
            f"{topk_key}={val_metrics[topk_key]:.4f} "
            f"best_val_acc={best_val_acc:.4f}"
        )

    # best model
    if best_state is not None:
        model.load_state_dict(best_state)

    return best_val_acc, criterion