set -e

echo "=============================================="
echo " TORCH Grad Accum Experiments"
echo " Models: bert, modernbert | Global BS: 64/256/1024"
echo " Epochs: 5"
echo " per-device batch_size: default.yaml (64)"
echo "=============================================="

run_one () {
  MODEL_NAME=$1
  GLOBAL_BS=$2
  ACCUM=$3
  LR=$4

  python main_torch.py \
    model=$MODEL_NAME \
    train.epochs=5 \
    train.grad_accum_steps=$ACCUM \
    optimizer.lr=$LR \
    wandb.project=nlp-study-exp2-lr
}

run_one bert 64 4 5e-5
run_one bert 256 16 2e-4
run_one bert 1024 64 8e-4

run_one modernbert 64 4 5e-5
run_one modernbert 256 16 2e-4
run_one modernbert 1024 64 8e-4

echo ""
echo "All TORCH experiments finished."