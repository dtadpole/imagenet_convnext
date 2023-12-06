import os
import math
import argparse
import wandb
import torch
from torch import optim, nn, utils, Tensor
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, DeviceStatsMonitor
from convnext import convnext_tiny, convnext_small, convnext_small_2, convnext_base, convnext_large, convnext_xlarge
from maxvit import max_vit_tiny_224, max_vit_small_224, max_vit_base_224, max_vit_large_224
from maxvit import MaxViT

wandb_project = "ImageNet"

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('-f', '--folder', default='./imagenet/',
                    help='path to dataset (default: ./imagenet/)')
parser.add_argument('-a', '--arch', default='ConvNeXt_T',
                    help='model arch (default: ConvNeXt_T)')
parser.add_argument('-b', '--batch_size', default=64, type=int,
                    help="batch size (default: 64)")
parser.add_argument('--epoch', default=60, type=int,
                    help="total epoch (default: 60)")
parser.add_argument('--warmup_epoch', default=5, type=float,
                    help='warmup epoch (default: 5)')
parser.add_argument('--finetune_epoch', default=10, type=float,
                    help='finetune epoch (default: 10)')
parser.add_argument('--lr', default=3e-4, type=float,
                    help="learning rate (default: 3e-4)")
parser.add_argument('--lr_end', default=1e-5, type=float,
                    help="ending learning rate (default: 1e-5)")
parser.add_argument('--drop_rate', default=0.1, type=float,
                    help="drop rate (default: 0.1)")
parser.add_argument('--drop_path_rate', default=0.1, type=float,
                    help="drop path rate (default: 0.1)")
parser.add_argument('--beta1', default=0.9, type=float,
                    help="beta1 (default: 0.9)")
parser.add_argument('--beta2', default=0.999, type=float,
                    help="beta2 (default: 0.999)")
parser.add_argument('--weight_decay', default=0.1, type=float,
                    help='weight decay (default: 0.1)')
parser.add_argument('--compile', default=False, type=bool,
                    help="compile model (default: False)")
parser.add_argument('--workers', default=5, type=int,
                    help="number of workers (default: 5)")
parser.add_argument('--prefetch', default=5, type=int,
                    help="number of prefetch (default: 5)")
parser.add_argument('--precision', default='bf16-mixed', type=str,
                    help='training precision (default: bf16-mixed)')
parser.add_argument('--transform_ops', default=2, type=int,
                    help='number of ops, default 2')
parser.add_argument('--transform_mag', default=15, type=int,
                    help="magnitude (default: 15)")
parser.add_argument('--use_grn', default=0, type=int,
                    help="use GRN (default: 0; available: 1, 2)")
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
    val_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, prefetch_factor=args.prefetch, pin_memory=True)


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
        self._loss = nn.CrossEntropyLoss()
        self.save_hyperparameters(args)
        self.validation_step_outputs = []
        self.wandb_inited = False

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        images, target = batch
        output = self._model(images)
        loss = self._loss(output, target)
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        # step lr scheduler
        sch = self.lr_schedulers()
        lr = sch.get_last_lr()[0]
        sch.step()
        self.log_dict({
            "train_loss": loss,
            "train_acc1": acc1,
            "train_acc5": acc5,
            "train_lr": lr,
        })
        return loss

    def validation_step(self, batch, batch_idx):
        # validation_step defines the validation loop.
        images, target = batch
        output = self._model(images)
        loss = self._loss(output, target)
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        log_dict = {
            "val_loss": loss,
            "val_acc1": acc1,
            "val_acc5": acc5,
        }
        self.validation_step_outputs.append(log_dict)

    def on_validation_epoch_end(self):
        # outs is a list of whatever stored in `validation_step`
        outs = self.validation_step_outputs
        loss = torch.stack([x['val_loss'] for x in outs]).mean()
        acc1 = torch.stack([x['val_acc1'] for x in outs]).mean()
        acc5 = torch.stack([x['val_acc5'] for x in outs]).mean()
        self.validation_step_outputs.clear()  # free memory
        # get lr
        sch = self.lr_schedulers()
        lr = sch.get_last_lr()[0]
        # log results
        log_dict = {
            "val_loss": loss,
            "val_acc1": acc1,
            "val_acc5": acc5,
            "val_lr": lr,
        }
        self.log_dict(log_dict, sync_dist=True)
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
