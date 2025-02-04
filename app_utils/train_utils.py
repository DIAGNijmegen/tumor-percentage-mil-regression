import torch.nn as nn
import torch
import os
import numpy as np
import torch.optim as optim
from torch.utils.data import DataLoader, Sampler, WeightedRandomSampler, RandomSampler, SequentialSampler, sampler
from functools import partial
from sklearn.metrics import roc_auc_score
from typing import Optional
from app_utils.loss import WeightedMSELoss, weighting_continuous_values
from topk.svm import SmoothTop1SVM

class SubsetSequentialSampler(Sampler):
    """Samples elements sequentially from a given list of indices, without replacement.
	Arguments:
		indices (sequence): a sequence of indices
	"""

    def __init__(self, indices):
        self.indices = indices

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)


class SchedulerFactory:
    def __init__(
            self,
            optimizer: torch.optim.Optimizer,
            params: Optional[dict] = None,
    ):

        self.scheduler = None
        self.name = params.name
        if self.name == "step":
            self.scheduler = optim.lr_scheduler.StepLR(
                optimizer, step_size=params.step_size, gamma=params.gamma
            )
        elif self.name == "cosine":
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=10
            )
        elif self.name == "reduce_lr_on_plateau":
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode=params.mode,
                factor=params.factor,
                patience=params.patience,
                min_lr=params.min_lr,
            )
        elif self.name:
            raise KeyError(f"{self.name} not supported")

    def get_scheduler(self):
        return self.scheduler


class EarlyStopping:
    def __init__(self, patience=20, stop_epoch=50, verbose=False, counter=0):

        self.patience = patience
        self.stop_epoch = stop_epoch
        self.verbose = verbose
        self.counter = counter
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf

    def __call__(self, epoch, val_loss, model, optimizer, ckpt_name='checkpoint.pt'):

        score = -val_loss
        self.epoch = epoch
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, optimizer, ckpt_name)
        elif score < self.best_score:
            self.counter += 1
            print(f'\nEarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience and epoch > self.stop_epoch:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, optimizer, ckpt_name)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, optimizer, ckpt_name):
        if self.verbose:
            print(f'\nValidation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        #torch.save(model.state_dict(), ckpt_name)
        torch.save({
            'epoch': self.epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': val_loss,
            'counter': self.counter,
            }, ckpt_name)
        self.val_loss_min = val_loss


def compute_auc(labels, preds):
    # Check if there are any cases with label 0
    if 0 not in labels:
        # No labels with 0, return None or handle as needed
        return None

    # Convert labels to binary (1 if not 0, else 0)
    bin_labels = [1 if el != 0 else 0 for el in labels]

    # Compute and return AUC
    return roc_auc_score(bin_labels, preds)


def define_loss(device, loss, n_classes, labels=None):
    if loss == 'svm':
        loss_fn = SmoothTop1SVM(n_classes=n_classes)
        if device.type == 'cuda':
            loss_fn = loss_fn.cuda()
    elif loss == 'ce':
        loss_fn = nn.CrossEntropyLoss()
    elif loss == 'l1':
        loss_fn = nn.SmoothL1Loss()
    elif loss == 'mse':
        loss_fn = nn.MSELoss()
    elif loss == 'balanced':
        weights = weighting_continuous_values(labels)
        loss_fn = WeightedMSELoss(weights)
    print('Done!')

    return loss_fn


def get_optim(model, cfg):
    print("Optimizer is: ", cfg.optimizer.name)
    if cfg.optimizer.name == "adam":
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=cfg.optimizer.lr,
                               weight_decay=cfg.optimizer.wd)
    elif cfg.optimizer.name == 'sgd':
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=cfg.optimizer.lr, momentum=0.9,
                              weight_decay=cfg.optimizer.wd)
    elif cfg.optimizer.name == 'adamw':
        optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=cfg.optimizer.lr,
                                weight_decay=cfg.optimizer.wd)
    elif cfg.optimizer.name == 'lbfgs':
        optimizer = optim.LBFGS(filter(lambda p: p.requires_grad, model.parameters()), lr=cfg.optimizer.lr)
    return optimizer


def get_simple_loader(dataset, split, cfg):
    batch_size = cfg.settings.batch_size
    if split == 'test':
        batch_size = 1
    if split == 'train':
        dl = DataLoader(dataset, batch_size=batch_size, sampler=RandomSampler(dataset), num_workers=cfg.settings.num_workers)
    else:
        dl = DataLoader(dataset, batch_size=batch_size, sampler=SequentialSampler(dataset), num_workers=cfg.settings.num_workers)
    return dl


def initialize_weights(module):
    for m in module.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            m.bias.data.zero_()

        elif isinstance(m, nn.BatchNorm1d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)


def make_weights_for_balanced_classes_split(dataset):
    N = float(len(dataset))
    weight_per_class = [N / len(dataset.slide_cls_ids[c]) for c in range(len(dataset.slide_cls_ids))]
    weight = [0] * int(N)
    for idx in range(len(dataset)):
        y = dataset.getlabel(idx)
        weight[idx] = weight_per_class[y]
    return torch.DoubleTensor(weight)


def print_network(net):
    num_params = 0
    num_params_train = 0
    print(net)

    for param in net.parameters():
        n = param.numel()
        num_params += n
        if param.requires_grad:
            num_params_train += n

    print('Total number of parameters: %d' % num_params)
    print('Total number of trainable parameters: %d' % num_params_train)


def seed_torch(device, seed=7):
    import random
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    
