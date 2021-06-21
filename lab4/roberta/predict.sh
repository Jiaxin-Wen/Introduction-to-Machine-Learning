echo "Using GPUs: [$1]"
echo "VERSION: [$VERSION]"

python roberta.py \
    --test_set ../data/test.json \
    --predict \
    --predict_out_path ../prediction_result/roberta_heu_version_$VERSION.csv \
    --load_dir ../logs/roberta/lightning_logs/version_$VERSION/checkpoints \
    --gpus $1 \
    --batch_size 100