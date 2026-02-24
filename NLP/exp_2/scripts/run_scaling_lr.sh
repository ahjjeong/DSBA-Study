set -e

echo "=============================================="
echo " TORCH Global BS + LR Scaling Experiments"
echo " Models: bert, modernbert | Global BS: 64/256/1024"
echo " Epochs: 5"
echo " per-device batch_size: default.yaml (16)"
echo " global_bs -> grad_accum_steps is computed in code"
echo " lr scaling: 64->5e-5, 256->2e-4, 1024->8e-4"
echo "=============================================="

run_one () {
  MODEL_NAME=$1
  GLOBAL_BS=$2
  LR=$3

  echo ""
  echo "----------------------------------------------"
  echo " Model=$MODEL_NAME | global_bs=$GLOBAL_BS | lr=$LR"
  echo "----------------------------------------------"

  python main_torch.py \
    backend=torch \
    model=$MODEL_NAME \
    train.epochs=5 \
    train.global_batch_size=$GLOBAL_BS \
    optimizer.lr=$LR \
    wandb.project=nlp-study-exp2-lr-again
}

run_one bert 64 5e-5
run_one bert 256 1e-4
run_one bert 1024 2e-4

run_one modernbert 64 5e-5
run_one modernbert 256 1e-4
run_one modernbert 1024 2e-4

echo ""
echo "All TORCH LR-scaling experiments finished."