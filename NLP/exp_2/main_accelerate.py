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

from accelerate import Accelerator

from src.model import EncoderForClassification
from src.data import get_dataloader
from src.utils import seed_everything, set_wandb


def batch_correct_total(logits: torch.Tensor, labels: torch.Tensor):
    preds = logits.argmax(dim=-1)
    correct = (preds == labels).sum()
    total = torch.tensor(labels.numel(), device=labels.device)
    return correct, total


@torch.no_grad()
def eval_loop(model, dataloader, accelerator: Accelerator):
    model.eval()

    loss_sum = 0.0
    n_sum = 0
    correct_sum = torch.tensor(0, device=accelerator.device)
    total_sum = torch.tensor(0, device=accelerator.device)

    for batch in dataloader:
        logits, loss = model(**batch)

        # loss 집계 (현재 프로세스 기준)
        bsz = batch["labels"].size(0)
        loss_sum += loss.item() * bsz
        n_sum += bsz

        # accuracy 집계 (모든 프로세스 데이터 수집)
        c, t = batch_correct_total(logits, batch["labels"])
        c = accelerator.gather_for_metrics(c)
        t = accelerator.gather_for_metrics(t)
        correct_sum += c.sum()
        total_sum += t.sum()

    loss_avg = loss_sum / max(n_sum, 1)
    acc = (correct_sum.float() / total_sum.float()).item() if total_sum.item() > 0 else 0.0
    return loss_avg, acc


@hydra.main(version_base=None, config_path="configs", config_name="default")
def main(configs: omegaconf.DictConfig):
    if os.path.split(os.getcwd())[-1] == "outputs":
        print(OmegaConf.to_yaml(configs))

    grad_accum_steps = int(getattr(configs.train, "grad_accum_steps", 1))
    accelerator = Accelerator(gradient_accumulation_steps=grad_accum_steps)
    device = accelerator.device

    seed_everything(int(getattr(configs, "seed", 42)))

    # model
    num_labels = int(getattr(configs.dataset, "num_labels", 2))
    model = EncoderForClassification(configs.model, num_labels=num_labels)

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

    # data
    train_loader = get_dataloader(configs, split="train", model_name=configs.model.name)
    valid_loader = get_dataloader(configs, split="valid", model_name=configs.model.name)
    test_loader = get_dataloader(configs, split="test", model_name=configs.model.name)

    # accelerator 준비
    model, optimizer, train_loader, valid_loader, test_loader = accelerator.prepare(
        model, optimizer, train_loader, valid_loader, test_loader
    )

    # wandb
    use_wandb = False
    if accelerator.is_main_process:
        use_wandb = set_wandb(configs)

    best_val_acc = -1.0
    global_step = 0

    run_dir = HydraConfig.get().runtime.output_dir
    checkpoint_path = os.path.join(run_dir, "best.pt")

    # train
    for epoch in range(int(configs.train.epochs)):
        model.train()

        train_loss_sum, n_train = 0.0, 0
        correct_sum, total_sum = torch.tensor(0, device=device), torch.tensor(0, device=device)

        win_loss_sum = torch.tensor(0.0, device=device)
        win_n = torch.tensor(0, device=device)
        win_correct = torch.tensor(0, device=device)
        win_total = torch.tensor(0, device=device)

        for batch in tqdm(train_loader, desc=f"Train [Epoch {epoch+1}]", disable=not accelerator.is_local_main_process):
            with accelerator.accumulate(model):
                logits, loss = model(**batch)
                accelerator.backward(loss)

                bsz = batch["labels"].size(0)
                win_loss_sum += loss.detach() * bsz
                win_n += bsz

                preds = logits.argmax(dim=-1)
                win_correct += (preds == batch["labels"]).sum()
                win_total += batch["labels"].numel()

                # step 단위 로그 (sync가 일어날 때만)
                if accelerator.sync_gradients:
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)

                    if use_wandb and accelerator.is_main_process:
                        red_loss_sum = accelerator.reduce(win_loss_sum, reduction="sum")
                        red_n = accelerator.reduce(win_n, reduction="sum")
                        red_correct = accelerator.reduce(win_correct, reduction="sum")
                        red_total = accelerator.reduce(win_total, reduction="sum")

                        loss_step = (red_loss_sum / red_n.clamp(min=1)).item()
                        acc_step = (red_correct.float() / red_total.clamp(min=1).float()).item()

                        wandb.log(
                            {"train/loss_step": loss_step, "train/acc_step": acc_step},
                            step=global_step,
                        )

                    win_loss_sum.zero_()
                    win_n.zero_()
                    win_correct.zero_()
                    win_total.zero_()

                    global_step += 1

            # stats (local)
            bsz = batch["labels"].size(0)
            train_loss_sum += loss.item() * bsz
            n_train += bsz

            # stats (gathered)
            c, t = batch_correct_total(logits, batch["labels"])
            c, t = accelerator.gather_for_metrics((c, t))
            correct_sum += c.sum()
            total_sum += t.sum()

        train_loss = train_loss_sum / max(n_train, 1)
        train_acc = (correct_sum.float() / total_sum.float()).item() if total_sum.item() > 0 else 0.0

        # valid
        val_loss, val_acc = eval_loop(model, valid_loader, accelerator)

        if accelerator.is_main_process:
            print(
                f"[Epoch {epoch+1}] "
                f"train_loss={train_loss:.4f}, train_acc={train_acc:.4f} | "
                f"val_loss={val_loss:.4f}, val_acc={val_acc:.4f}"
            )

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
            # unwrap을 통해 DDP 래퍼가 제거된 순수 모델 상태 저장
            unwrapped_model = accelerator.unwrap_model(model)
            accelerator.save(unwrapped_model.state_dict(), checkpoint_path)

    # test
    # 모든 프로세스가 저장 완료를 기다림
    accelerator.wait_for_everyone()

    # best model 불러오기
    if os.path.exists(checkpoint_path):
        state_dict = torch.load(checkpoint_path, map_location=device)
        accelerator.unwrap_model(model).load_state_dict(state_dict)
        if accelerator.is_main_process:
            print(f"Loaded best model from {checkpoint_path} for testing.")

    test_loss, test_acc = eval_loop(model, test_loader, accelerator)
    
    if accelerator.is_main_process:
        print(f"[Final Test] loss={test_loss:.4f}, acc={test_acc:.4f}")
        if use_wandb:
            wandb.log({"test/loss": test_loss, "test/acc": test_acc}, step=global_step)


if __name__ == "__main__":
    main()