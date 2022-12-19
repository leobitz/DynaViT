import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.loggers import TensorBoardLogger

import torch
import torchvision
import warmup_scheduler
from timm.data import Mixup
from utils import create_model
from timm.optim import create_optimizer
from timm.scheduler import create_scheduler

import argx
from agent import Baseline, Skipper
from augment import new_data_aug_generator
from datasets import build_dataset
from helpers import get_criterion

import models_v2

class Net(pl.LightningModule):
    def __init__(self, hparams):
        super(Net, self).__init__()

        self.hparams.update(vars(hparams))

        self.model = create_model(
                    hparams.model,
                    pretrained=hparams.finetune,
                    num_classes=hparams.nb_classes,
                    drop_rate=hparams.drop,
                    drop_path_rate=hparams.drop_path,
                    drop_block_rate=None,
                    img_size=hparams.input_size,
                    dynamic=False,
                    finetune_num_classes=hparams.nb_classes,
                    rl_dropout = hparams.rl_dropout
                )

        self.mixup_fn = None
        mixup_active = hparams.mixup > 0 or hparams.cutmix > 0. or hparams.cutmix_minmax is not None
        if mixup_active:
            self.mixup_fn = Mixup(
                mixup_alpha=hparams.mixup, cutmix_alpha=hparams.cutmix, cutmix_minmax=hparams.cutmix_minmax,
                prob=hparams.mixup_prob, switch_prob=hparams.mixup_switch_prob, mode=hparams.mixup_mode,
                label_smoothing=hparams.smoothing, num_classes=hparams.nb_classes)

        self.criterion = get_criterion(args, mixup_active)
        self.eval_criterion =  torch.nn.CrossEntropyLoss()
        # self.automatic_optimization = False
        # self.grad_acc = hparams.grad_acc
        self.baseline_mse_loss = torch.nn.MSELoss()

    def configure_optimizers(self):

        # self.optimizer = create_optimizer(self.hparams, self.model)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.hparams.lr, betas=(
            self.hparams.beta1, self.hparams.beta2), weight_decay=self.hparams.weight_decay)

        self.base_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=self.hparams.max_epochs, eta_min=self.hparams.min_lr)
        self.lr_scheduler = warmup_scheduler.GradualWarmupScheduler(
            self.optimizer, multiplier=1., total_epoch=self.hparams.warmup_epochs, 
                    after_scheduler=self.base_scheduler)
        # self.lr_scheduler = self.base_scheduler

        return ([self.optimizer], [self.lr_scheduler])

    def forward(self, x):
        return self.model(x)
        
    def training_step(self, batch, batch_idx):
        img, label = batch
        # labelx = label.unsqueeze(-1)

        if self.mixup_fn:
            img, labelx = self.mixup_fn(img, label)

        out = self(img)
        loss = self.criterion(out, labelx)

        raw_acc = torch.eq(out.detach().argmax(-1), label).float()
        acc = raw_acc.mean()
        
        self.log("loss", loss)
        self.log("acc", acc)

        return loss

    
    def validation_step(self, batch, batch_idx):
        img, label = batch
        # labelx = label.unsqueeze(-1)
        out = self(img)
        # print(out.shape, label.shape)
        loss = self.criterion(out, label))
        raw_acc = torch.eq(out.detach().argmax(-1), label).float()
        acc = raw_acc.mean()

        self.log("val_loss", loss)
        self.log("val_acc", acc)

        return loss
        
parser = argx.get_args_parser()

# parser = DistillModule.add_model_specific_args(parser)
parser = pl.Trainer.add_argparse_args(parser)

args = parser.parse_args()

torch.manual_seed(args.seed)
np.random.seed(args.seed)


dataset_train, args.nb_classes = build_dataset(is_train=True, args=args)
dataset_val, _ = build_dataset(is_train=False, args=args)

# sampler_train = torch.utils.data.RandomSampler(dataset_train)
# sampler_val = torch.utils.data.SequentialSampler(dataset_val)

data_loader_train = torch.utils.data.DataLoader(
        dataset_train, 
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True,
        pin_memory=args.pin_mem,
        drop_last=True,
    )
# if args.ThreeAugment:
#     data_loader_train.dataset.transform = new_data_aug_generator(args)

data_loader_val = torch.utils.data.DataLoader(
    dataset_val, shuffle=False,
    batch_size=int(1.5 * args.batch_size),
    num_workers=args.num_workers,
    pin_memory=args.pin_mem,
    drop_last=False
)

net = Net(args)


cbs = []
# logger = TensorBoardLogger("logs", name=args.exp_name)
logger = WandbLogger(name=args.exp_name, project=args.proj_name)
trainer = pl.Trainer.from_argparse_args(args, 
                                logger=logger,
				                strategy='ddp', 
                                callbacks=cbs)
        
trainer.fit(model=net, train_dataloaders=data_loader_train, val_dataloaders=data_loader_val)
