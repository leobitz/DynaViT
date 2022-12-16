

#<<<<<<< HEAD
python dyna-train.py --model deit_small_patch16_LS  --data-path data/tinyimagenet/tiny-imagenet-200 --batch-size 64 --lr 4e-3  --weight-decay 0.05 --sched cosine --input-size 224 --eval-crop-ratio 1.0 --reprob 0.0  --smoothing 0.0 --warmup-epochs 5 --drop 0.0  --seed 0 --opt adamw --warmup-lr 1e-6 --mixup .8 --drop-path 0.05 --cutmix 1.0 --unscale-lr  --color-jitter 0.3 --ThreeAugment --proj_name tinyimagenet --exp_name initial --num_workers 8 --max_epochs 128 --gpus 4 --grad-acc 8  --dynamic --data-set TINYIMNET
#=======
#python dyna-train.py --model deit_small_patch16_LS  --data-path data/tinyimagenet/tiny-imagenet-200 --batch-size 256 --lr 4e-3 --epochs 800 --weight-decay 0.05 --sched cosine --input-size 224 --eval-crop-ratio 1.0 --reprob 0.0  --smoothing 0.0 --warmup-epochs 5 --drop 0.0  --seed 0 --opt adamw --warmup-lr 1e-6 --mixup .8 --drop-path 0.05 --cutmix 1.0 --unscale-lr  --color-jitter 0.3 --ThreeAugment --proj_name tinyimagenet --exp_name initial --num_workers 0 --max_epochs 10 --gpus 2  --dynamic --data-set TINYIMNET
#>>>>>>> refs/remotes/origin/main

