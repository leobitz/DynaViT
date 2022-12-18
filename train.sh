
# python static-train.py --model deit_small_patch16_LS  \
# --data-path data/tinyimagenet/tiny-imagenet-200 \
# --batch-size 64 --lr 0.001  --weight-decay 0.05 \
# --sched cosine --input-size 224 --eval-crop-ratio 1.0 \
# --reprob 0.0  --smoothing 0.1 --warmup-epochs 0 --drop 0.0  \
# --seed 0 --opt adamw --warmup-lr 1e-6 --mixup .8 --drop-path 0.1 \
# --cutmix 1.0 --unscale-lr  --color-jitter 0.3 --ThreeAugment \
# --proj_name 'tinyimagenet' --exp_name 'initial-static' --num_workers 8 \
# --max_epochs 128 --accelerator 'gpu' --devices 4 --grad-acc 1  --data-set 'TINYIMNET' \
# --rl-lr 0.0001 --rl-dropout 0.0 --rl-gamma 0.9 --dynamic --finetune

python static-train.py --model deit_tiny_patch16_LS  \
--data-path data/cifar10 \
--batch-size 512 --lr 0.00001  --weight-decay 5e-5 \
--input-size 32 \
--reprob 0.0  --smoothing 0.0 --warmup-epochs 5 --drop 0.0  \
--seed 0 --opt adamw --warmup-lr 1e-6 --mixup 0.8 --drop-path 0.1 \
--cutmix 1.0  --clip-grad 1.0 \
--proj_name 'tinyimagenet' --exp_name 'initial-static-cfar10' --num_workers 8 \
--max_epochs 128 --accelerator 'gpu' --devices 1 --grad-acc 1  --data-set 'CIFAR10' \
--rl-lr 0.0001 --rl-dropout 0.0 --rl-gamma 0.9 --dynamic 