set -e

echo "=============================="
echo " Run 1: BERT-base-uncased"
echo "=============================="

python main.py \
  model=bert

echo ""
echo "=============================="
echo " Run 2: ModernBERT-base"
echo "=============================="

python main.py \
  model=modernbert

echo ""
echo "All experiments finished."