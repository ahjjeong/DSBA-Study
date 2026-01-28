import os
import hydra
from omegaconf import DictConfig, OmegaConf

from src.utils import seed_everything, get_device, append_csv, ensure_dir
from src.wandb_utils import init_wandb
from src.data import get_loaders
from src.models import build_model
from src.train_eval import train, evaluate

@hydra.main(config_path="configs", config_name="default.yaml", version_base=None)
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))

    seed_everything(int(cfg.seed))
    device = get_device(str(cfg.device))

    run = init_wandb(cfg)

    train_loader, val_loader, test_loader, num_classes, sizes = get_loaders(
        cfg_dataset=cfg.dataset,
        cfg_data=cfg.data,
        cfg_model=cfg.model,
        seed=int(cfg.seed),
    )
    print("Dataset sizes:", sizes)

    model = build_model(cfg.model, num_classes=num_classes).to(device)
    
    topk = 5
    topk_key = f"top{topk}_err"
    test_topk_key = f"test_{topk_key}"

    best_val_acc, criterion = train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        cfg_train=cfg.train,
        cfg_opt=cfg.optimizer,
        device=device,
        wandb_run=run,
        k=topk,
    )

    test_metrics = evaluate(
        model=model,
        loader=test_loader,
        device=device,
        criterion=criterion,
        k=topk,
    )

    # wandb summary
    if run is not None:
        run.summary["model"] = str(cfg.model.name)
        run.summary["pretrained"] = bool(cfg.model.pretrained)
        run.summary["train_fraction"] = float(cfg.data.train_fraction)
        run.summary["test_acc"] = float(test_metrics["acc"])
        run.summary["test_top1_err"] = float(test_metrics["top1_err"])
        run.summary[test_topk_key] = float(test_metrics.get(topk_key, 0.0))
        run.finish()

    # CSV 저장
    out_dir = os.path.abspath(cfg.out.dir)
    ensure_dir(out_dir)
    csv_path = os.path.join(out_dir, cfg.out.results_csv)

    header = [
        "dataset", "model", "pretrained", "train_fraction", "train_size",
        "best_val_acc", "test_acc", "test_top1_err", test_topk_key
    ]
    row = [
        str(cfg.dataset.name),
        str(cfg.model.name),
        int(bool(cfg.model.pretrained)),
        float(cfg.data.train_fraction),
        int(sizes["train"]),
        float(best_val_acc),
        float(test_metrics["acc"]),
        float(test_metrics["top1_err"]),
        float(test_metrics.get(topk_key, 0.0)),
    ]
    append_csv(csv_path, header, row)

    print(f"Saved CSV -> {csv_path}")
    print(
        f"FINAL: model={cfg.model.name} pretrained={cfg.model.pretrained} "
        f"frac={cfg.data.train_fraction} "
        f"test_acc={test_metrics['acc']:.4f} "
        f"top1_err={test_metrics['top1_err']:.4f} "
        f"{topk_key}={test_metrics.get(topk_key, 0.0):.4f}"
    )

if __name__ == "__main__":
    main()