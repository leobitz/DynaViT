
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.scheduler import create_scheduler
from timm.optim import create_optimizer
from timm.utils import NativeScaler, get_state_dict, ModelEma
import torch

class LabelSmoothingCrossEntropyLoss(torch.nn.Module):
    def __init__(self, classes, smoothing=0.0, dim=-1):
        super(LabelSmoothingCrossEntropyLoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.cls = classes
        self.dim = dim

    def forward(self, pred, target):
        if len(pred) == 3:
            pred = pred[0]
        pred = pred.log_softmax(dim=self.dim)
        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.cls - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        return torch.mean(torch.sum(-true_dist * pred, dim=self.dim))    


def get_criterion(args, mixup_active):
    if mixup_active:
        # smoothing is handled with mixup label transform
        # criterion = LabelSmoothingCrossEntropyLoss(10, args.smoothing)
        # criterion = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
        criterion = torch.nn.CrossEntropyLoss()
    elif args.smoothing:
        criterion = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
    else:
        criterion = torch.nn.CrossEntropyLoss()
        
    if args.bce_loss:
        criterion = torch.nn.BCEWithLogitsLoss()

    # criterion = torch.nn.CrossEntropyLoss()
    return criterion