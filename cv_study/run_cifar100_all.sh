# =========================
# CIFAR-100 experiment
# =========================

set -e

EXP_NAME="cifar100"
OUT_DIR="outputs/results_${EXP_NAME}"
CSV_NAME="results_${EXP_NAME}.csv"

WANDB_PROJECT="cv-study-e2"
WANDB_GROUP="${EXP_NAME}"

MODEL_CONFIGS=(
  resnet50_scratch
  resnet50_im1k
  vit_s16_scratch
  vit_s16_im1k
)

FRACTION=1.0

echo "==========================================="
echo "[run_cifar100_all.sh] CIFAR-100 (fraction=${FRACTION})"
echo "Models configs: ${MODEL_CONFIGS[*]}"
echo "OUT_DIR: ${OUT_DIR}"
echo "CSV: ${CSV_NAME}"
echo "WANDB_PROJECT: ${WANDB_PROJECT}"
echo "WANDB_GROUP: ${WANDB_GROUP}"
echo "==========================================="

for mcfg in "${MODEL_CONFIGS[@]}"; do
  RUN_NAME="${EXP_NAME}_${mcfg}"

  echo ""
  echo "-------------------------------------------"
  echo "[RUN] dataset=cifar100 model=${mcfg}"
  echo "-------------------------------------------"

  python main.py \
    dataset=cifar100 \
    model="${mcfg}" \
    data.train_fraction="${FRACTION}" \
    out.dir="${OUT_DIR}" \
    out.results_csv="${CSV_NAME}" \
    wandb.project="${WANDB_PROJECT}" \
    wandb.group="${WANDB_GROUP}" \
    wandb.name="${RUN_NAME}" \
    wandb.tags="[cifar100]"

  echo "[DONE] ${RUN_NAME}"
  echo "-------------------------------------------"
done

echo ""
echo "==========================================="
echo "[run_cifar100_all.sh] All experiments finished."
echo "==========================================="