import os
import math
import argparse
import wandb
import torch
from torch import optim, nn, utils, Tensor
import torch.distributed as dist
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, DeviceStatsMonitor
from convnext import convnext_tiny, convnext_small, convnext_small_2, convnext_base, convnext_large, convnext_xlarge
from maxvit import max_vit_tiny_224, max_vit_small_224, max_vit_base_224, max_vit_large_224
from maxvit import MaxViT
from timm.data.mixup import Mixup
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy

wandb_project = "ImageNet"

# basic params
parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('-f', '--folder', default='./imagenet/',
                    help='path to dataset (default: ./imagenet/)')
parser.add_argument('-a', '--arch', default='ConvNeXt_T',
                    help='model arch (default: ConvNeXt_T)')
parser.add_argument('-b', '--batch_size', default=64, type=int,
                    help="batch size (default: 64)")

# epoch and lr
parser.add_argument('--epoch', default=60, type=int,
                    help="total epoch (default: 60)")
parser.add_argument('--warmup_epoch', default=5, type=float,
                    help='warmup epoch (default: 5)')
parser.add_argument('--finetune_epoch', default=10, type=float,
                    help='finetune epoch (default: 10)')
parser.add_argument('--lr', default=1e-3, type=float,
                    help="learning rate (default: 1e-3)")
parser.add_argument('--lr_end', default=3e-5, type=float,
                    help="ending learning rate (default: 3e-5)")

# drop rate
parser.add_argument('--drop_rate', default=0.1, type=float,
                    help="drop rate (default: 0.1)")
parser.add_argument('--drop_path_rate', default=0.2, type=float,
                    help="drop path rate (default: 0.2)")
parser.add_argument('--beta1', default=0.9, type=float,
                    help="beta1 (default: 0.9)")
parser.add_argument('--beta2', default=0.999, type=float,
                    help="beta2 (default: 0.999)")
parser.add_argument('--weight_decay', default=0.1, type=float,
                    help='weight decay (default: 0.1)')

# workers
parser.add_argument('--compile', default=False, type=bool,
                    help="compile model (default: False)")
parser.add_argument('--precision', default='bf16-mixed', type=str,
                    help='training precision (default: bf16-mixed)')
parser.add_argument('--workers', default=5, type=int,
                    help="number of workers (default: 5)")
parser.add_argument('--prefetch', default=8, type=int,
                    help="number of prefetch (default: 8)")

# Augmentation parameters
parser.add_argument('--color_jitter', type=float, default=0.4, metavar='PCT',
                    help='Color jitter factor (default: 0.4)')
parser.add_argument('--aa', type=str, default='rand-m9-mstd0.5-inc1', metavar='NAME',
                    help='Use AutoAugment policy. "v0" or "original". " + "(default: rand-m9-mstd0.5-inc1)'),
parser.add_argument('--smoothing', type=float, default=0.1,
                    help='Label smoothing (default: 0.1)')
parser.add_argument('--train_interpolation', type=str, default='bicubic',
                    help='Training interpolation (random, bilinear, bicubic default: "bicubic")')

# * Mixup params
parser.add_argument('--mixup', type=float, default=0.8,
                    help='mixup alpha, mixup enabled if > 0.')
parser.add_argument('--cutmix', type=float, default=1.0,
                    help='cutmix alpha, cutmix enabled if > 0.')
parser.add_argument('--cutmix_minmax', type=float, nargs='+', default=None,
                    help='cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)')
parser.add_argument('--mixup_prob', type=float, default=1.0,
                    help='Probability of performing mixup or cutmix when either/both is enabled')
parser.add_argument('--mixup_switch_prob', type=float, default=0.5,
                    help='Probability of switching to cutmix when both mixup and cutmix enabled')
parser.add_argument('--mixup_mode', type=str, default='batch',
                    help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"')

# transforms
parser.add_argument('--transform_ops', default=2, type=int,
                    help='number of ops, default 2')
parser.add_argument('--transform_mag', default=10, type=int,
                    help="magnitude (default: 10)")
args = parser.parse_args()

torch.set_float32_matmul_precision('medium')


def build_model(arch="ConvNeXt_T"):
    if arch.lower() == "ConvNeXt_T".lower():
        return convnext_tiny(drop_path_rate=args.drop_path_rate)
    elif arch == "ConvNeXt_S".lower():
        return convnext_small(drop_path_rate=args.drop_path_rate)
    elif arch == "ConvNeXt_S2".lower():
        return convnext_small_2(drop_path_rate=args.drop_path_rate)
    elif arch.lower() == "ConvNeXt_B".lower():
        return convnext_base(drop_path_rate=args.drop_path_rate)
    elif arch.lower() == "ConvNeXt_L".lower():
        return convnext_large(drop_path_rate=args.drop_path_rate)
    elif arch.lower() == "ConvNeXt_XL".lower():
        return convnext_xlarge(drop_path_rate=args.drop_path_rate)
    elif arch.lower() == "MaxViT_T".lower():
        return max_vit_tiny_224(drop=args.drop_rate, drop_path=args.drop_path_rate)
    elif arch.lower() == "MaxViT_S".lower():
        return max_vit_small_224(drop=args.drop_rate, drop_path=args.drop_path_rate)
    elif arch.lower() == "MaxViT_B".lower():
        return max_vit_base_224(drop=args.drop_rate, drop_path=args.drop_path_rate)
    elif arch.lower() == "MaxViT_L".lower():
        return max_vit_large_224(drop=args.drop_rate, drop_path=args.drop_path_rate)
    else:
        raise Exception('Unknown arch %s' % arch)


# mixup
mixup_fn = None
mixup_active = args.mixup > 0 or args.cutmix > 0. or args.cutmix_minmax is not None
if mixup_active:
    print("Mixup is activated!")
    mixup_fn = Mixup(
        mixup_alpha=args.mixup, cutmix_alpha=args.cutmix, cutmix_minmax=args.cutmix_minmax,
        prob=args.mixup_prob, switch_prob=args.mixup_switch_prob, mode=args.mixup_mode,
        label_smoothing=args.smoothing)

# train dataset
train_dataset = datasets.ImageFolder(
    os.path.join(args.folder, 'train'),
    transforms.Compose([
        transforms.RandAugment(num_ops=args.transform_ops,
                               magnitude=args.transform_mag),
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ]))

val_dataset = datasets.ImageFolder(
    os.path.join(args.folder, 'val'),
    transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ]))

train_loader = utils.data.DataLoader(
    train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, prefetch_factor=args.prefetch, pin_memory=True)
val_loader = utils.data.DataLoader(
    val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, prefetch_factor=args.prefetch, pin_memory=True)


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


class Model(L.LightningModule):

    def __init__(self):
        super().__init__()
        self._model = build_model(args.arch)
        if mixup_fn is not None:
            self._train_loss_fn = SoftTargetCrossEntropy()
        elif args.smoothing > 0.:
            self._train_loss_fn = LabelSmoothingCrossEntropy(
                smoothing=args.smoothing)
        else:
            self._train_loss_fn = nn.CrossEntropyLoss()
        self._eval_loss_fn = nn.CrossEntropyLoss()
        self.save_hyperparameters(args)
        self.train_step_outputs = []
        self.validation_step_outputs = []
        self.wandb_inited = False

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        images, targets = batch
        if mixup_fn is not None:
            mixup_images, mixup_targets = mixup_fn(images, targets)
            mixup_output = self._model(mixup_images)
            loss = self._train_loss_fn(mixup_output, mixup_targets)
            with torch.no_grad():
                output = self._model(images)
                loss_raw = self._eval_loss_fn(output, targets)
        else:
            output = self._model(images)
            loss = self._train_loss_fn(output, targets)
            loss_raw = loss
        # accuracy
        acc1, acc5 = accuracy(output, targets, topk=(1, 5))
        # step lr scheduler
        sch = self.lr_schedulers()
        lr = sch.get_last_lr()[0]
        sch.step()
        log_dict = {
            "train_loss": loss,
            "train_loss_raw": loss_raw,
            "train_acc1": acc1,
            "train_acc5": acc5,
            "train_lr": lr,
        }
        self.train_step_outputs.append(log_dict)
        self.log_dict(log_dict)
        return loss

    def validation_step(self, batch, batch_idx):
        # validation_step defines the validation loop.
        images, targets = batch
        output = self._model(images)
        loss = self._eval_loss_fn(output, targets)
        acc1, acc5 = accuracy(output, targets, topk=(1, 5))
        log_dict = {
            "val_loss": loss,
            "val_acc1": acc1,
            "val_acc5": acc5,
        }
        self.validation_step_outputs.append(log_dict)
        self.log_dict(log_dict, sync_dist=True)

    def on_validation_epoch_end(self):
        # val_outs is a list of whatever stored in `validation_step`
        val_outs = self.validation_step_outputs
        val_loss = torch.stack([x['val_loss'] for x in val_outs]).mean()
        val_acc1 = torch.stack([x['val_acc1'] for x in val_outs]).mean()
        val_acc5 = torch.stack([x['val_acc5'] for x in val_outs]).mean()
        self.validation_step_outputs.clear()  # free val memory
        # train_outs is a list of whatever stored in `train_step`
        train_outs = self.train_step_outputs
        train_loss = torch.stack([x['train_loss'] for x in train_outs]).mean() if len(
            train_outs) > 0 else val_loss
        train_acc1 = torch.stack([x['train_acc1'] for x in train_outs]).mean() if len(
            train_outs) > 0 else val_acc1
        train_acc5 = torch.stack([x['train_acc5'] for x in train_outs]).mean() if len(
            train_outs) > 0 else val_acc5
        self.train_step_outputs.clear()  # free train memory
        # all_gather
        tensorized = torch.Tensor([
            train_loss,
            train_acc1,
            train_acc5,
            val_loss,
            val_acc1,
            val_acc5
        ]).cuda()
        gather_t = [torch.ones_like(tensorized)
                    for _ in range(dist.get_world_size())]
        dist.all_gather(gather_t, tensorized)
        result_t = torch.mean(torch.stack(gather_t), dim=0)
        # get lr
        sch = self.lr_schedulers()
        lr = sch.get_last_lr()[0]
        # log results
        log_dict = {
            "train_loss": result_t[0],
            "train_acc1": result_t[1],
            "train_acc5": result_t[2],
            "val_loss": result_t[3],
            "val_acc1": result_t[4],
            "val_acc5": result_t[5],
            "train_lr": lr,
        }
        if self.trainer.local_rank == 0:
            if not self.wandb_inited:
                model_name = type(self._model).__name__
                param_count = sum(p.numel() for p in self._model.parameters())
                wandb_name = model_name + '__' + f"{param_count:_}"
                wandb.init(project=wandb_project, name=wandb_name, config=args)
                self.wandb_inited = True
            wandb.log(log_dict)

    def configure_optimizers(self):
        iters_per_epoch = self._num_iters_per_epoch()
        print('iters_per_epoch', iters_per_epoch)
        optimizer = optim.AdamW(
            self.parameters(), lr=args.lr, betas=(args.beta1, args.beta2), weight_decay=args.weight_decay)
        warmup_scheduler = LinearLR(optimizer,
                                    start_factor=1e-4,
                                    end_factor=1.0,
                                    total_iters=math.ceil(iters_per_epoch*args.warmup_epoch))
        main_scheduler = CosineAnnealingLR(optimizer,
                                           iters_per_epoch *
                                           (args.epoch-args.warmup_epoch -
                                            args.finetune_epoch),
                                           eta_min=args.lr_end)
        finetune_scheduler = LinearLR(optimizer,
                                      start_factor=args.lr_end/args.lr,
                                      end_factor=args.lr_end/args.lr,
                                      total_iters=math.ceil(iters_per_epoch*args.finetune_epoch))
        scheduler = SequentialLR(optimizer,
                                 schedulers=[
                                     warmup_scheduler,
                                     main_scheduler,
                                     finetune_scheduler],
                                 milestones=[
                                     math.ceil(iters_per_epoch *
                                               args.warmup_epoch),
                                     math.ceil(iters_per_epoch*(args.epoch-args.finetune_epoch))])
        return [optimizer], [scheduler]

    def _num_iters_per_epoch(self) -> float:
        """Total training steps inferred from datamodule and devices."""
        dataset_size = len(train_loader)
        devices = max(1, self.trainer.num_devices)
        return dataset_size / devices


model = Model()
if args.compile:
    model = torch.compile(model)


checkpoint_callback = ModelCheckpoint(
    save_top_k=3, monitor="val_acc1", mode="max", filename="model-{epoch:02d}-{val_acc1:.2f}-{val_loss:.2f}")

trainer = L.Trainer(limit_train_batches=None, max_epochs=args.epoch, profiler="simple",
                    precision=args.precision, callbacks=[DeviceStatsMonitor(), checkpoint_callback])
trainer.fit(model=model, train_dataloaders=train_loader,
            val_dataloaders=val_loader)
