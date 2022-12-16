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
                    pretrained=True,
                    num_classes=1000,
                    drop_rate=hparams.drop,
                    drop_path_rate=hparams.drop_path,
                    drop_block_rate=None,
                    img_size=hparams.input_size,
                    dynamic=hparams.dynamic,
                    finetune_num_classes=hparams.nb_classes
                )
        self.skipper = Skipper(input_size=self.model.hidden_size, 
                               hidden_size=self.model.hidden_size,
                               n_layers=self.model.num_layers, # depth
                               dropout=hparams.rl_dropout)
        self.baseline = Baseline(input_size=self.model.hidden_size,
                                 hidden_size=self.model.hidden_size,
                                 n_layers=self.model.num_layers,
                                 dropout=hparams.rl_dropout)

        self.mixup_fn = None
        mixup_active = hparams.mixup > 0 or hparams.cutmix > 0. or hparams.cutmix_minmax is not None
        if mixup_active:
            self.mixup_fn = Mixup(
                mixup_alpha=hparams.mixup, cutmix_alpha=hparams.cutmix, cutmix_minmax=hparams.cutmix_minmax,
                prob=hparams.mixup_prob, switch_prob=hparams.mixup_switch_prob, mode=hparams.mixup_mode,
                label_smoothing=hparams.smoothing, num_classes=hparams.nb_classes)

        self.criterion = get_criterion(args, mixup_active)
        # self.automatic_optimization = False
        self.baseline_mse_loss = torch.nn.MSELoss()
        self.cache = {}

    def configure_optimizers(self):

        self.optimizer = create_optimizer(self.hparams, self.model)

        self.base_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=self.hparams.max_epochs, eta_min=self.hparams.min_lr)
        self.lr_scheduler = warmup_scheduler.GradualWarmupScheduler(
            self.optimizer, multiplier=1., total_epoch=self.hparams.warmup_epochs, 
                    after_scheduler=self.base_scheduler)

        self.rl_optimizer = torch.optim.AdamW(self.skipper.parameters(), lr=self.hparams.rl_lr, betas=(
            self.hparams.beta1, self.hparams.beta2), weight_decay=5e-5)

        self.baseline_optimizer = torch.optim.AdamW(self.baseline.parameters(), lr=self.hparams.rl_lr, betas=(
            self.hparams.beta1, self.hparams.beta2), weight_decay=5e-5)

        self.rl_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.rl_optimizer, T_max=self.hparams.max_epochs, eta_min=self.hparams.min_lr)

        self.baseline_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.baseline_optimizer, T_max=self.hparams.max_epochs, eta_min=self.hparams.min_lr)

        return ([self.optimizer, self.rl_optimizer, self.baseline_optimizer], 
                [self.lr_scheduler, self.rl_scheduler, self.baseline_scheduler])

    def forward(self, x, skipper, baseline):
        return self.model(x, skipper, baseline)
        
    def training_step(self, batch, batch_idx, optimizer_idx):
        img, label = batch
        print(self.model.device, self.skipper.device, self.baseline.device)
        device_name = torch.cuda.get_device_name(img.device)
        if optimizer_idx == 0:
            labelx = label.unsqueeze(-1)
            bsizes = []
            # print(img.device, label.device)
            if self.mixup_fn:
                img, labelx = self.mixup_fn(img, labelx)

            out = self(img, self.skipper, self.baseline)
            outx, bsizes, n_layer_proc, log_actions, state_values = out
            loss = self.criterion(outx, labelx)

            n_layers = self.model.num_layers
            raw_acc = torch.eq(outx.detach().argmax(-1), label).float()
            acc = raw_acc.mean()

            bsizesx = np.array(bsizes) / len(outx)
            proc_ratio = bsizesx.mean()

            rewards = raw_acc - self.hparams.proc_alpha * \
                n_layer_proc - self.hparams.reward_epsilon
            total_reward = rewards

            Gs = [rewards]
            for xi in range(n_layers - 2, -1, -1):

                total_reward = total_reward * self.hparams.rl_gamma
                Gs.append(total_reward)

            Gs.reverse()

            Gs = torch.stack(Gs).transpose(0, 1)
            Gs = (Gs - Gs.mean(dim=1, keepdim=True))/Gs.std(dim=1, keepdim=True)

            bs = torch.stack(state_values).squeeze().transpose(0, 1)
            log_actions = torch.stack(log_actions).transpose(0, 1)

            self.cache[device_name+'bs'] = bs
            self.cache[device_name+'gs'] = Gs
            self.cache[device_name+'log_actions'] = log_actions
            self.log("loss", loss)
            self.log("acc", acc)
            self.log("reward", rewards.mean())
            self.log("proc_ratio", proc_ratio)

            for ibs, bs in enumerate(bsizes):
                self.log(f"layer-{ibs+1}", float(bs)/len(raw_acc))
        # print(Gs.shape, bs.shape)
        elif optimizer_idx == 2:
            bs = self.cache[device_name+'bs']
            Gs = self.cache[device_name+'gs']
            baseline_loss = self.baseline_mse_loss(bs, Gs)
            self.log("baseline_loss", baseline_loss)
        elif optimizer_idx == 1:
            bs = self.cache[device_name+'bs']
            Gs = self.cache[device_name+'gs']
            log_actions = self.cache[device_name+'log_actions']
            delta = Gs - bs.clone().detach()

            policy_loss = (-delta * log_actions).sum(axis=1).mean()
            self.log("rl_loss", policy_loss)
        # opt, rl_optim, bs_optim = self.optimizers()

        # classification loss gradient step
        # opt.zero_grad()
        # self.manual_backward(loss)
        # opt.step()

        # # baseline loss gradient step
        # bs_optim.zero_grad()
        # self.manual_backward(baseline_loss)
        # bs_optim.step()

        # policy loss gradient step
        
        # rl_optim.zero_grad()
        # self.manual_backward(policy_loss)
        # rl_optim.step()

        
        
        
        

        return loss

    # def training_epoch_end(self, training_step_outputs):

    #     sch, rl_sch, bs_sch = self.lr_schedulers()
    #     sch.step()
    #     bs_sch.step()
    #     rl_sch.step()

    
    def validation_step(self, batch, batch_idx):
        img, label = batch
        out = self(img, self.skipper, self.baseline)

        out, bsizes, n_layer_proc, log_actions, state_values = out

        bsizesx = np.array(bsizes) / len(out)
        proc_ratio = bsizesx.mean()  # / (n_layers - 2)

        # print((-label.unsqueeze(-1) * torch.log_softmax(out, axis=-1)).shape)

        loss = self.criterion(out, label.unsqueeze(-1))
        raw_acc = torch.eq(out.detach().argmax(-1), label).float()
        acc = raw_acc.mean()

        rewards = raw_acc - n_layer_proc  # + 1) / (n_layers) )
        self.log("val_reward", rewards.mean())
        self.log("val_proc_ratio", proc_ratio)
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

sampler_train = torch.utils.data.RandomSampler(dataset_train)
sampler_val = torch.utils.data.SequentialSampler(dataset_val)

data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )
if args.ThreeAugment:
    data_loader_train.dataset.transform = new_data_aug_generator(args)

data_loader_val = torch.utils.data.DataLoader(
    dataset_val, sampler=sampler_val,
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
				strategy='dp', 
                                callbacks=cbs)
        
trainer.fit(model=net, train_dataloaders=data_loader_train, val_dataloaders=data_loader_val)
