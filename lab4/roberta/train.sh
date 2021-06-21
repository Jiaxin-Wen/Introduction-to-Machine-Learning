echo "Using GPUs: [$1]"


python roberta.py \
    --train_set ../data/train.json \
    --valid_set ../data/valid.json \
    --save_dir ../logs/roberta \
    --lr 3e-5 \
    --max_epochs 10 \
    --gpus $1 \
    --seed 43675123 \
    --batch_size 64