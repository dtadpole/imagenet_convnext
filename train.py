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
from model import ConvNeXt

wandb_project = "ConvNeXt"
wandb_name = "Tiny-28.6M"


parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('-f', '--folder', default='./imagenet/',
                    help='path to dataset (default: ./imagenet/)')
parser.add_argument('--epoch', default=90, type=int,
                    help="total epoch (default: 90)")
parser.add_argument('--warmup-epoch', default=5, type=float,
                    help='warmup epoch (default: 5)')
parser.add_argument('--lr', default=3e-4, type=float,
                    help="learning rate (default: 3e-4)")
parser.add_argument('--lr_end', default=1e-5, type=float,
                    help="ending learning rate (default: 1e-5)")
parser.add_argument('--beta1', default=0.9, type=float,
                    help="beta1 (default: 0.9)")
parser.add_argument('--beta2', default=0.999, type=float,
                    help="beta2 (default: 0.999)")
parser.add_argument('--weight_decay', default=0.1, type=float,
                    help='weight decay (default: 0.1)')
parser.add_argument('--batch_size', default=64, type=int,
                    help="batch size (default: 64)")
parser.add_argument('--compile', default=False, type=bool,
                    help="compile model (default: False)")
parser.add_argument('--workers', default=5, type=int,
                    help="number of workers (default: 5)")
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
    train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, prefetch_factor=5, pin_memory=True)
val_loader = utils.data.DataLoader(
    val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, prefetch_factor=5, pin_memory=True)


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
        self._model = ConvNeXt(depths=[3, 3, 9, 3], dims=[96, 192, 384, 768])
        self._loss = nn.CrossEntropyLoss()
        self.save_hyperparameters(args)
        wandb.init(project=wandb_project, name=wandb_name, config=args)
        self.iters_per_epoch = -1

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        images, target = batch
        output = self._model(images)
        loss = self._loss(output, target)
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        optimizer = self.optimizers()
        lr = optimizer.param_groups[0]['lr']
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
        self.log_dict(log_dict)
        wandb.log(log_dict)
        return loss, acc1, acc5

    def configure_optimizers(self):
        if self.iters_per_epoch < 0:
            self.iters_per_epoch = self._num_training_steps()
        optimizer = optim.AdamW(
            self.parameters(), lr=args.lr, betas=(args.beta1, args.beta2), weight_decay=args.weight_decay)
        main_scheduler = CosineAnnealingLR(
            optimizer, self.iters_per_epoch*(args.epoch-args.warmup_epoch), eta_min=args.lr_end)
        warmup_scheduler = LinearLR(optimizer, start_factor=1e-4,
                                    end_factor=1.0, total_iters=math.ceil(self.iters_per_epoch*args.warmup_epoch))
        scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, main_scheduler],
                                milestones=[math.ceil(self.iters_per_epoch*args.warmup_epoch)])
        return [optimizer], [scheduler]

    def _num_training_steps(self) -> float:
        """Total training steps inferred from datamodule and devices."""
        dataset = train_loader
        if self.trainer.max_steps:
            return self.trainer.max_steps

        dataset_size = (
            self.trainer.limit_train_batches
            if self.trainer.limit_train_batches != 0
            else len(dataset)
        )

        num_devices = max(1, self.trainer.num_gpus, self.trainer.num_processes)
        if self.trainer.tpu_cores:
            num_devices = max(num_devices, self.trainer.tpu_cores)

        effective_batch_size = dataset.batch_size * self.trainer.accumulate_grad_batches * num_devices
        return (dataset_size / effective_batch_size) * self.trainer.max_epochs


model = Model()
if args.compile:
    model = torch.compile(model)


checkpoint_callback = ModelCheckpoint(
    save_top_k=3, monitor="val_acc1", mode="max", filename="model-{epoch:02d}-{val_acc1:.2f}")

trainer = L.Trainer(limit_train_batches=None, max_epochs=args.epoch, profiler="simple",
                    precision=args.precision, callbacks=[DeviceStatsMonitor(), checkpoint_callback])
trainer.fit(model=model, train_dataloaders=train_loader,
            val_dataloaders=val_loader)
