export CUDA_VISIBLE_DEVICES=0
python main.py \
    --model_name AnomalyTransformer \
    --default_cfg ./configs/default_setting.yaml \
    --model_cfg ./configs/model_setting.yaml \
    --opts \
    DATASET.dataname=PSM \
    DATASET.batch_size=256 \
    DATASET.seq_len=100 \
    TRAIN.epoch=3 \
    TRAIN.early_stopping_count=3 \
    MODELSETTING.anomaly_ratio=1.0
