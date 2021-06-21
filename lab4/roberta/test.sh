echo "Using GPUs: [$1]"
echo "VERSION: [$VERSION]"

python roberta.py \
    --test_set ../data/valid.json \
    --test \
    --load_dir ../logs/roberta/lightning_logs/version_$VERSION/checkpoints \
    --gpus $1 \
    --batch_size 256