#!/usr/bin/env bash
set -e

model_name=iTransformer
seq_len=96
batch_size=32

default_cfg=./configs/default_setting.yaml
model_cfg=./configs/model_setting.yaml

data_name=ETTh1
data_path=./dataset/ETT-small
freq=h
c_in=7

run_one () {
  pred_len=$1
  d_model=$2
  d_ff=$3

  accelerate launch main.py \
    --model_name $model_name \
    --default_cfg $default_cfg \
    --model_cfg $model_cfg \
    DATASET.seq_len $seq_len \
    DATASET.pred_len $pred_len \
    DATASET.datadir $data_path \
    DATASET.dataname $data_name \
    TRAIN.batch_size $batch_size \
    DEFAULT.exp_name forecasting_${data_name}_${seq_len}_${pred_len} \
    MODELSETTING.e_layers 2 \
    MODELSETTING.d_model $d_model \
    MODELSETTING.d_ff $d_ff \
    MODELSETTING.enc_in $c_in \
    MODELSETTING.dec_in $c_in \
    MODELSETTING.c_out $c_in \
    MODELSETTING.freq $freq
}

run_one 96  256 256
run_one 192 256 256
run_one 336 512 512
run_one 720 512 512