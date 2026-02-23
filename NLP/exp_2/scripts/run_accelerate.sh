set -e

echo "=============================================="
echo " ACCELERATE Grad Accum Experiments"
echo " Models: bert, modernbert | Global BS: 64/256/1024"
echo " Epochs: 5"
echo " per-device batch_size: default.yaml (64)"
echo "=============================================="

run_one () {
  MODEL_NAME=$1
  GLOBAL_BS=$2
  ACCUM=$3

  echo ""
  echo "----------------------------------------------"
  echo " Model=$MODEL_NAME | global_bs=$GLOBAL_BS | accum=$ACCUM"
  echo "----------------------------------------------"

  accelerate launch main_accelerate.py -- \
    backend=accelerate \
    model=$MODEL_NAME \
    train.epochs=5 \
    train.grad_accum_steps=$ACCUM
}

run_one bert 64 4
run_one bert 256 16
run_one bert 1024 64

run_one modernbert 64 4
run_one modernbert 256 16
run_one modernbert 1024 64

echo ""
echo "All ACCELERATE experiments finished."