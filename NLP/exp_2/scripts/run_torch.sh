set -e

echo "=============================================="
echo " TORCH Global Batch Size Experiments"
echo " Models: bert, modernbert | Global BS: 64/256/1024"
echo " Epochs: 5"
echo " per-device batch_size: default.yaml (16)"
echo " global_bs -> grad_accum_steps is computed in code"
echo "=============================================="

run_one () {
  MODEL_NAME=$1
  GLOBAL_BS=$2

  echo ""
  echo "----------------------------------------------"
  echo " Model=$MODEL_NAME | global_bs=$GLOBAL_BS"
  echo "----------------------------------------------"

  python main_torch.py \
    backend=torch \
    model=$MODEL_NAME \
    train.epochs=5 \
    train.global_batch_size=$GLOBAL_BS
}

run_one bert 64
run_one bert 256
run_one bert 1024

run_one modernbert 64
run_one modernbert 256
run_one modernbert 1024

echo ""
echo "All TORCH experiments finished."