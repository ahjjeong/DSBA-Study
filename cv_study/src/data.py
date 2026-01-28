import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


def pick_norm_stats(cfg_dataset, cfg_norm, cfg_model):
    '''
    정규화에 사용할 mean/std 선택
    - pretrained=True  -> imagenet
    - pretrained=False -> 해당 dataset
    '''
    pretrained = bool(cfg_model.pretrained) # pretrain 여부
    key = "imagenet" if pretrained else cfg_dataset.name

    # key에 해당하는 mean/std 값이 없을 경우
    if key not in cfg_norm:
        available = ", ".join(cfg_norm.keys()) # 가능한 목록 출력
        raise KeyError(
            f"[Normalization Error] '{key}' normalization stats not found.\n"
            f"Available options: {available}\n"
            f"Check configs/data.normalize in default.yaml."
        )

    mean = cfg_norm[key]["mean"]
    std = cfg_norm[key]["std"]
    return tuple(mean), tuple(std)


def build_transforms(cfg_dataset, cfg_data, cfg_model):
    '''
    학습/평가 단계에서 사용할 이미지 전처리
    '''
    img_size = int(cfg_data.img_size)
    
    mean, std = pick_norm_stats(
        cfg_dataset=cfg_dataset,
        cfg_norm=cfg_data.normalize,
        cfg_model=cfg_model,
    )

    # 학습용 transform
    train_tf = transforms.Compose([
        transforms.Resize(img_size), # 입력 해상도 통일
        transforms.RandomCrop(img_size, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    # 평가용 transform
    eval_tf = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    return train_tf, eval_tf


def stratified_train_val_split(targets: np.ndarray, val_size: int, seed: int):
    '''
    클래스 비율을 유지(stratified)하면서
    전체 데이터를 train / validation으로 분할하는 함수
    '''
    indices = np.arange(len(targets))
    sss = StratifiedShuffleSplit(
        n_splits=1,
        test_size=val_size,
        random_state=seed,
    )
    train_idx, val_idx = next(sss.split(indices, targets))
    return train_idx, val_idx


def train_subsample(
    indices: np.ndarray,
    targets: np.ndarray,
    fraction: float,
    seed: int,
    stratified: bool,
):
    '''
    train-size scaling 실험을 위해 training data를 subsampling하는 함수
    indices: 현재 train pool의 원본 인덱스
    targets: indices에 해당하는 label 배열 (stratified를 위해 필요)
    '''
    # fraction이 1 이상이면 전체 데이터 사용
    if fraction >= 1.0:
        return indices

    n = len(indices)
    k = max(1, int(round(n * fraction)))

    # stratified하게 sampling
    if stratified:
        sss = StratifiedShuffleSplit(n_splits=1, train_size=k, random_state=seed)
        sub_idx, _ = next(sss.split(np.arange(n), targets))
        return indices[sub_idx]
    # random하게 sampling
    else:
        rng = np.random.default_rng(seed)
        chosen = rng.choice(indices, size=k, replace=False)
        return chosen


def get_loaders(cfg_dataset, cfg_data, cfg_model, seed: int):
    """
    데이터셋을
    -> train/val/test로 나누고
    → DataLoader로 감싸서
    모델 학습에 쓸 수 있는 형태로 만드는 함수
    """
    pretrained = bool(cfg_model.pretrained)

    # 이미지 전처리
    train_tf, eval_tf = build_transforms(
        cfg_dataset=cfg_dataset,
        cfg_data=cfg_data,
        cfg_model=cfg_model,
    )

    # 데이터셋 다운로드
    root = str(cfg_dataset.path)
    # CIFAR-10일 경우
    if cfg_dataset.name.lower() == "cifar10":
        base_train = datasets.CIFAR10(root=root, train=True, download=True, transform=train_tf)
        base_val = datasets.CIFAR10(root=root, train=True, download=False, transform=eval_tf)
        test_ds = datasets.CIFAR10(root=root, train=False, download=True, transform=eval_tf)
        targets = np.array(base_train.targets)
    # CIFAR-100일 경우
    elif cfg_dataset.name.lower() == "cifar100":
        base_train = datasets.CIFAR100(root=root, train=True, download=True, transform=train_tf)
        base_val = datasets.CIFAR100(root=root, train=True, download=False, transform=eval_tf)
        test_ds = datasets.CIFAR100(root=root, train=False, download=True, transform=eval_tf)
        targets = np.array(base_train.targets)
    # 없는 데이터셋일 경우
    else:
        raise NotImplementedError(
            f"Dataset '{cfg_dataset.name}' is not supported yet."
        )

    # train/val split
    val_size = int(cfg_data.val_size)
    # stratified하게 split
    if bool(cfg_data.stratified_split):
        train_idx, val_idx = stratified_train_val_split(targets, val_size=val_size, seed=seed)
    # random하게 split
    else:
        rng = np.random.default_rng(seed)
        perm = rng.permutation(len(targets))
        val_idx = perm[:val_size]
        train_idx = perm[val_size:]

    # Scale 실험 - training subsampling
    train_targets_pool = targets[train_idx]

    train_idx = train_subsample(
        indices=train_idx,
        targets=train_targets_pool,
        fraction=float(cfg_data.train_fraction),
        seed=seed,
        stratified=bool(cfg_data.stratified_fraction),
    )

    train_ds = Subset(base_train, train_idx.tolist())
    val_ds = Subset(base_val, val_idx.tolist())

    # DataLoader
    batch_size = int(cfg_data.batch_size)
    num_workers = int(cfg_data.num_workers)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    # 모델 build할 때 필요
    num_classes = int(cfg_dataset.num_classes)

    # tracking용
    sizes = {
        "train": len(train_ds),
        "val": len(val_ds),
        "test": len(test_ds),
    }
    return train_loader, val_loader, test_loader, num_classes, sizes