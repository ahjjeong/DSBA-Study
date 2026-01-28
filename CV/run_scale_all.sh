# =========================
# Train fraction scaling experiment (CIFAR-10)
# =========================

set -e

EXP_NAME="scale"
OUT_DIR="outputs/results_${EXP_NAME}"
CSV_NAME="results_${EXP_NAME}.csv"

WANDB_PROJECT="cv-study-e1"
WANDB_GROUP="${EXP_NAME}"

MODEL_CONFIGS=(
  resnet50_scratch
  resnet50_im1k
  vit_s16_scratch
  vit_s16_im1k
)

FRACTIONS=(1.0 0.5 0.2 0.1)

echo "==========================================="
echo "[run_scale_all.sh] Fraction scaling experiment (CIFAR-10)"
echo "EXP_NAME: ${EXP_NAME}"
echo "Model configs: ${MODEL_CONFIGS[*]}"
echo "Fractions: ${FRACTIONS[*]}"
echo "OUT_DIR: ${OUT_DIR}"
echo "CSV: ${CSV_NAME}"
echo "WANDB_PROJECT: ${WANDB_PROJECT}"
echo "WANDB_GROUP: ${WANDB_GROUP}"
echo "==========================================="

for mcfg in "${MODEL_CONFIGS[@]}"; do
  for frac in "${FRACTIONS[@]}"; do

    RUN_NAME="${EXP_NAME}_${mcfg}_frac${frac}"

    echo ""
    echo "-------------------------------------------"
    echo "[RUN] dataset=cifar10 model=${mcfg}, train_fraction=${frac}"
    echo "      wandb: project=${WANDB_PROJECT}, group=${WANDB_GROUP}, name=${RUN_NAME}"
    echo "      out: dir=${OUT_DIR}, csv=${CSV_NAME}"
    echo "-------------------------------------------"

    python main.py \
      dataset=cifar10 \
      model="${mcfg}" \
      data.train_fraction="${frac}" \
      out.dir="${OUT_DIR}" \
      out.results_csv="${CSV_NAME}" \
      wandb.project="${WANDB_PROJECT}" \
      wandb.group="${WANDB_GROUP}" \
      wandb.name="${RUN_NAME}"

    echo "[DONE] ${RUN_NAME}"
    echo "-------------------------------------------"
  done
done

echo ""
echo "==========================================="
echo "[run_scale_all.sh] All experiments finished."
echo "==========================================="