import wandb

def init_wandb(cfg):
    if not bool(cfg.wandb.enable):
        return None

    run_name = cfg.wandb.name
    if run_name is None:
        run_name = (
            f"{cfg.dataset.name}_"
            f"{cfg.model.name}_pre{int(cfg.model.pretrained)}_"
            f"frac{cfg.data.train_fraction}_seed{cfg.seed}"
        )

    group = getattr(cfg.wandb, "group", None)
    if group is None:
        group = f"{cfg.dataset.name}-scale"

    run = wandb.init(
        project=cfg.wandb.project,
        entity=cfg.wandb.entity,
        name=run_name,
        group=group,
        tags=list(cfg.wandb.tags) if cfg.wandb.tags is not None else None,
        mode=cfg.wandb.mode,
        save_code=bool(cfg.wandb.save_code),
        config={
            "seed": int(cfg.seed),
            "device": str(cfg.device),

            "dataset": str(cfg.dataset.name),
            "model": str(cfg.model.name),
            "pretrained": bool(cfg.model.pretrained),

            "img_size": int(cfg.data.img_size),
            "batch_size": int(cfg.data.batch_size),
            "num_workers": int(cfg.data.num_workers),
            "val_size": int(cfg.data.val_size),
            "train_fraction": float(cfg.data.train_fraction),

            "epochs": int(cfg.train.epochs),
            "warmup_epochs": int(cfg.train.warmup_epochs),
            "loss_function": str(cfg.train.loss_function),
            "amp": bool(cfg.train.amp),

            "optimizer": str(cfg.optimizer.name),
            "opt_lr": float(cfg.optimizer.lr),
            "opt_weight_decay": float(getattr(cfg.optimizer, "weight_decay", 0.0)),
        },
    )
    return run