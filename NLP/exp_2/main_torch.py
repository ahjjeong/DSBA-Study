import os
os.environ["TORCHINDUCTOR_DISABLE"] = "1"
os.environ["TORCH_COMPILE_DISABLE"] = "1"

import torch
import hydra
import wandb
from tqdm import tqdm
import omegaconf
from omegaconf import OmegaConf

from hydra.core.hydra_config import HydraConfig

from src.model import EncoderForClassification
from src.data import get_dataloader
from src.utils import seed_everything, get_device, set_wandb


def calculate_accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    preds = logits.argmax(dim=-1)
    correct = (preds == labels).sum().item()
    return correct / labels.size(0)


def valid_iter(model, batch, device):
    batch = {k: v.to(device) for k, v in batch.items()}
    logits, loss = model(**batch)
    acc = calculate_accuracy(logits, batch["labels"])
    return loss.item(), acc


@hydra.main(version_base=None, config_path="configs", config_name="default")
def main(configs: omegaconf.DictConfig):
    print(OmegaConf.to_yaml(configs))

    seed_everything(int(getattr(configs, "seed", 42)))
    device = get_device(configs.device)

    # 모델 생성
    num_labels = int(getattr(configs.dataset, "num_labels", 2)) # IMDB가 이진분류이기 때문
    model = EncoderForClassification(configs.model, num_labels=num_labels).to(device)

    # 데이터 로더 생성
    train_loader = get_dataloader(configs, split="train", model_name=configs.model.name)
    valid_loader = get_dataloader(configs, split="valid", model_name=configs.model.name)
    test_loader = get_dataloader(configs, split="test", model_name=configs.model.name)

    # optimizer 생성
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

    # grad accumulation 설정
    per_device_bs = int(configs.dataset.batch_size)
    global_bs = int(configs.train.global_batch_size)

    if global_bs % per_device_bs != 0:
        raise ValueError(
            f"global_batch_size ({global_bs}) must be divisible by "
            f"per_device_batch_size ({per_device_bs})"
        )

    grad_accum_steps = global_bs // per_device_bs

    # wandb 켜기
    use_wandb = set_wandb(configs)

    best_val_acc = -1.0
    global_step = 0

    # best.pt 저장 경로
    run_dir = HydraConfig.get().runtime.output_dir
    checkpoint_path = os.path.join(run_dir, "best.pt")

    # train loop
    for epoch in range(int(configs.train.epochs)):
        model.train() # train 모드로 전환
        optimizer.zero_grad(set_to_none=True) # gradient 초기화
        
        # epoch 단위 통계 변수 초기화
        train_loss_sum, train_acc_sum, n_train = 0.0, 0.0, 0

        # 누적 단위 통계 변수 초기화
        win_loss_sum, win_correct, win_total = 0.0, 0, 0

        # micro-batch 단위 학습
        for step, batch in enumerate(tqdm(train_loader, desc=f"Train [Epoch {epoch+1}]"), start=1):
            batch = {k: v.to(device) for k, v in batch.items()} # 배치를 GPU로
            # forward
            logits, loss = model(**batch) 
            acc = calculate_accuracy(logits, batch["labels"])
            bsz = batch["labels"].size(0) # bsz = 16
            # 누적 단위 통계 업데이트
            win_loss_sum += loss.item() * bsz
            win_correct += (logits.argmax(dim=-1) == batch["labels"]).sum().item()
            win_total += bsz

            # accumulate gradients - 각 micro-batch의 gradient를 1/grad_accum_steps만큼 줄여서 누적
            (loss / grad_accum_steps).backward()

            # epoch 단위 통계 업데이트
            loss_item = loss.item()
            train_loss_sum += loss_item * bsz
            train_acc_sum += acc * bsz
            n_train += bsz

            # 가중치 업데이트 (grad_accum_steps마다)
            if step % grad_accum_steps == 0:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

                # wandb 로깅
                if use_wandb:
                    # 누적 구간 평균으로 wandb 로그
                    loss_step = win_loss_sum / max(win_total, 1)
                    acc_step = win_correct / max(win_total, 1)

                    wandb.log(
                        {"train/loss_step": loss_step, "train/acc_step": acc_step},
                        step=global_step,
                    )

                # 누적 단위 통계 초기화
                win_loss_sum = 0.0
                win_correct = 0
                win_total = 0

                global_step += 1

        # 나머지 gradient 처리 (배치 수가 누적 단계로 나누어 떨어지지 않을 때)
        if (len(train_loader) % grad_accum_steps) != 0:
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            # wandb 로깅
            if use_wandb and win_total > 0:
                loss_step = win_loss_sum / win_total
                acc_step = win_correct / win_total
                wandb.log(
                    {"train/loss_step": loss_step, "train/acc_step": acc_step},
                    step=global_step,
                )

            # 누적 단위 통계 초기화
            win_loss_sum = 0.0
            win_correct = 0
            win_total = 0

            global_step += 1

        # epoch 단위 통계 계산
        train_loss = train_loss_sum / n_train
        train_acc = train_acc_sum / n_train

        # validation
        model.eval() # eval 모드로 전환
        val_loss_sum, val_acc_sum, n_val = 0.0, 0.0, 0

        with torch.no_grad():
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

        # wandb 로깅
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

        # best 모델 저장
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), checkpoint_path)

    # test
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    test_loss_sum, test_acc_sum, n_test = 0.0, 0.0, 0
    with torch.no_grad():
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
            {"test/loss": test_loss, "test/acc": test_acc},
            step=global_step,
        )


if __name__ == "__main__":
    main()