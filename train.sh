
python dyna-train.py --model deit_small_patch16_LS  \
--data-path data/tinyimagenet/tiny-imagenet-200 \
--batch-size 64 --lr 0.00001  --weight-decay 0.05 \
--sched cosine --input-size 224 --eval-crop-ratio 1.0 \
--reprob 0.0  --smoothing 0.0 --warmup-epochs 0 --drop 0.0  \
--seed 0 --opt adamw --warmup-lr 1e-6 --mixup .8 --drop-path 0.05 \
--cutmix 1.0 --unscale-lr  --color-jitter 0.3 --ThreeAugment \
--proj_name tinyimagenet --exp_name initial --num_workers 8 \
--max_epochs 128 --gpus 4 --grad-acc 1  --data-set TINYIMNET \
--rl-lr 0.0001 --rl-dropout 0.0 --rl-gamma 0.9 --dynamic
