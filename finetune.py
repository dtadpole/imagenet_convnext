import os
import math
import argparse
import wandb
from copy import deepcopy
import torch
from torch import optim, nn, utils, Tensor
from torch.utils.data import DataLoader
import torch.distributed as dist
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
import lightning as L
from lightning.pytorch.strategies import DDPStrategy
from lightning.pytorch.callbacks import ModelCheckpoint, DeviceStatsMonitor, LearningRateMonitor, GradientAccumulationScheduler
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from train import PreTrainModule, EMA, build_mixup_fn, accuracy

wandb_project = "ImageNet"


def parse_finetune_args():
    # basic params
    parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
    parser.add_argument('-f', '--folder', default='./imagenet/',
                        help='path to dataset (default: ./imagenet/)')
    parser.add_argument('-a', '--arch', default='ConvNeXt_T',
                        help='model arch (default: ConvNeXt_T)')
    parser.add_argument('-b', '--batch_size', default=64, type=int,
                        help="batch size (default: 64)")
    parser.add_argument('-t', '--finetune', required=True, type=str,
                        help="finetune checkpoint path (required: True)")

    # epoch and lr
    parser.add_argument('--epoch', default=30, type=int,
                        help="total epoch (default: 30)")
    parser.add_argument('--lr', default=3e-5, type=float,
                        help="learning rate (default: 3e-5)")
    parser.add_argument('--accumulate_grad', default=1, type=int,
                        help="accumulate gradient (default: 1)")
    parser.add_argument('--gradient_clipping', default=1.0, type=float,
                        help="gradient clipping (default: 1.0)")
    parser.add_argument('--reference_batch_size', default=512, type=int,
                        help="reference batch size (default: 512)")

    # drop rate
    parser.add_argument('--drop_rate', default=0.1, type=float,
                        help="drop rate (default: 0.1)")
    parser.add_argument('--drop_path_rate', default=0.2, type=float,
                        help="drop path rate (default: 0.2)")
    parser.add_argument('--beta1', default=0.9, type=float,
                        help="beta1 (default: 0.9)")
    parser.add_argument('--beta2', default=0.999, type=float,
                        help="beta2 (default: 0.999)")
    parser.add_argument('--weight_decay', default=1e-8, type=float,
                        help='weight decay (default: 1e-8)')
    parser.add_argument('--model_ema_decay_1', default=None, type=float,
                        help='model ema decay (default: None)')
    parser.add_argument('--model_ema_decay_2', default=None, type=float,
                        help='model ema decay (default: None)')
    parser.add_argument('--model_ema_decay_3', default=None, type=float,
                        help='model ema decay (default: None)')
    parser.add_argument('--model_ema_decay_4', default=None, type=float,
                        help='model ema decay (default: None)')
    parser.add_argument('--model_ema_decay_5', default=None, type=float,
                        help='model ema decay (default: None)')
    # parser.add_argument('--ema_decay_eval', default=0.9999, type=float,
    #                     help='eval model ema decay (default: 0.9999)')
    # parser.add_argument('--ema_decay_train', default=0.999, type=float,
    #                     help='train model ema decay (default: 0.999)')

    # workers
    parser.add_argument('--compile', default=False, type=bool,
                        help="compile model (default: False)")
    parser.add_argument('--precision', default='bf16-mixed', type=str,
                        help='training precision (default: bf16-mixed)')
    parser.add_argument('--workers', default=5, type=int,
                        help="number of workers (default: 5)")
    parser.add_argument('--prefetch', default=10, type=int,
                        help="number of prefetch (default: 10)")

    # Mixup params
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
    parser.add_argument('--smoothing', type=float, default=0.1,
                        help='Label smoothing (default: 0.1)')

    # transforms
    parser.add_argument('--transform_ops', default=2, type=int,
                        help='number of ops, default 2')
    parser.add_argument('--transform_mag', default=15, type=int,
                        help="magnitude (default: 15)")
    args = parser.parse_args()
    return args


def build_data_loader(args):
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

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        prefetch_factor=args.prefetch,
        persistent_workers=True,
        pin_memory=True)

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        prefetch_factor=args.prefetch,
        persistent_workers=True,
        pin_memory=True)

    return train_loader, val_loader


# model_ema_eval: EMA = None
# model_ema_train: EMA = None


class FinetuneModule(L.LightningModule):

    def __init__(self, args, train_loader, val_loader):
        super().__init__()
        self._args = args
        self.train_loader = train_loader
        self.val_loader = val_loader
        self._mixup_fn = build_mixup_fn(args)
        checkpoint = PreTrainModule.load_from_checkpoint(
            args.finetune, args=args, train_loader=train_loader, val_loader=val_loader)
        self._model = checkpoint._model
        self._model_emas = nn.ModuleList()
        self.train_step_outputs = []
        self.validation_step_outputs = []
        if self._args.model_ema_decay_1 is not None:
            self._model_emas.append(
                EMA(self._model, decay=self._args.model_ema_decay_1))
            self.validation_step_outputs.append([])
        if self._args.model_ema_decay_2 is not None:
            self._model_emas.append(
                EMA(self._model, decay=self._args.model_ema_decay_2))
            self.validation_step_outputs.append([])
        if self._args.model_ema_decay_3 is not None:
            self._model_emas.append(
                EMA(self._model, decay=self._args.model_ema_decay_3))
            self.validation_step_outputs.append([])
        if self._args.model_ema_decay_4 is not None:
            self._model_emas.append(
                EMA(self._model, decay=self._args.model_ema_decay_4))
            self.validation_step_outputs.append([])
        if self._args.model_ema_decay_5 is not None:
            self._model_emas.append(
                EMA(self._model, decay=self._args.model_ema_decay_5))
            self.validation_step_outputs.append([])
        # self._train_loss_fn = checkpoint._train_loss_fn
        # self._eval_loss_fn = checkpoint._eval_loss_fn
        if self._mixup_fn is not None:
            self._train_loss_fn = SoftTargetCrossEntropy()
            self._eval_loss_fn = nn.CrossEntropyLoss()
        elif args.smoothing > 0.:
            self._train_loss_fn = LabelSmoothingCrossEntropy(
                smoothing=args.smoothing)
            self._eval_loss_fn = nn.CrossEntropyLoss()
        else:
            self._train_loss_fn = nn.CrossEntropyLoss()
            self._eval_loss_fn = self._train_loss_fn
        self.save_hyperparameters(args)
        self.wandb_inited = False
        self._profiled = False

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        images, targets = batch
        if self._mixup_fn is not None:
            mixup_images, mixup_targets = self._mixup_fn(images, targets)
            mixup_output = self._model(mixup_images)
            loss = self._train_loss_fn(mixup_output, mixup_targets)
            # with torch.no_grad():
            # output = self._model(images)
            # loss_raw = self._eval_loss_fn(output, targets)
            output = mixup_output
        else:
            output = self._model(images)
            loss = self._train_loss_fn(output, targets)
            # loss_raw = loss
        # accuracy
        acc1, acc5 = accuracy(output, targets, topk=(1, 5))
        # log
        self.log_dict({
            "train_loss": loss,
            "train_acc1": acc1,
            "train_acc5": acc5,
            "train_lr": self._args.lr,
        })
        # buffer
        self.train_step_outputs.append({
            "train_loss": loss,
            "train_acc1": acc1,
            "train_acc5": acc5,
            "train_lr": self._args.lr,
        })
        # return
        return loss

    def on_train_epoch_end(self):
        # keeps previous epoch info
        print(" ", end="")

    def optimizer_step(
        self,
        epoch,
        batch_idx,
        optimizer,
        optimizer_closure,
    ):
        optimizer.step(closure=optimizer_closure)
        # update ema model
        for model_ema in self._model_emas:
            model_ema.update(self._model)
        self._model_ema.update(self._model)

    def validation_step(self, batch, batch_idx):
        images, targets = batch
        # validation_step defines the validation loop.
        for idx, model_ in enumerate([self._model, *self._model_emas]):
            if idx == 0:
                output = model_(images)
            else:
                output = model_.module(images)
            loss = self._eval_loss_fn(output, targets)
            acc1, acc5 = accuracy(output, targets, topk=(1, 5))
            if idx == 0:
                log_dict = {
                    f"val_loss": loss,
                    f"val_acc1": acc1,
                    f"val_acc5": acc5,
                }
            else:
                log_dict = {
                    f"val_loss_{idx}": loss,
                    f"val_acc1_{idx}": acc1,
                    f"val_acc5_{idx}": acc5,
                }
            self.validation_step_outputs[idx].append(log_dict)
            self.log_dict(log_dict, sync_dist=True, rank_zero_only=False)

    def on_validation_epoch_end(self):
        # train_outs is a list of whatever stored in `train_step`
        log_data = []
        train_outs = self.train_step_outputs
        train_loss = torch.stack([x['train_loss'] for x in train_outs]).mean() if len(
            train_outs) > 0 else 0
        train_acc1 = torch.stack([x['train_acc1'] for x in train_outs]).mean() if len(
            train_outs) > 0 else 0
        train_acc5 = torch.stack([x['train_acc5'] for x in train_outs]).mean() if len(
            train_outs) > 0 else 0
        self.train_step_outputs.clear()  # free train memory
        log_data = [train_loss, train_acc1, train_acc5]
        for idx, _ in enumerate([self._model, *self._model_emas]):
            # val_outs is a list of whatever stored in `validation_step`
            val_outs = self.validation_step_outputs[idx]
            val_loss = torch.stack(
                [x[f'val_loss_{idx}' if idx != 0 else 'val_loss'] for x in val_outs]).mean()
            val_acc1 = torch.stack(
                [x[f'val_acc1_{idx}' if idx != 0 else 'val_acc1'] for x in val_outs]).mean()
            val_acc5 = torch.stack(
                [x[f'val_acc5_{idx}' if idx != 0 else 'val_acc5'] for x in val_outs]).mean()
            self.validation_step_outputs[idx].clear()  # free val memory
            log_data = log_data + [val_loss, val_acc1, val_acc5]
        # return if sanity checking
        if self.trainer.sanity_checking:
            return
        # all_gather
        tensorized = torch.Tensor(log_data).cuda()
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
            "train_lr": lr,
        }
        i = 3
        while (i < len(result_t)):
            if i == 3:
                log_dict.update({
                    "val_loss": result_t[i],
                    "val_acc1": result_t[i+1],
                    "val_acc5": result_t[i+2],
                })
            else:
                log_dict.update({
                    f"val_loss_{i//3-1}": result_t[i],
                    f"val_acc1_{i//3-1}": result_t[i+1],
                    f"val_acc5_{i//3-1}": result_t[i+2],
                })
            i += 3
        if self.trainer.local_rank == 0:
            if not self.wandb_inited:
                model_name = type(self._model).__name__
                param_count = sum(p.numel()
                                  for p in self._model.parameters())
                wandb_name = model_name + '__' + f"{param_count:_}"
                wandb.init(project=wandb_project, name=wandb_name,
                           group="PreTrain", config=self._args)
                self.wandb_inited = True
            wandb.log(log_dict)

    def configure_optimizers(self):
        _, _, effective_lr = self._steps_per_epoch()
        optimizer = optim.AdamW(
            # self.parameters(),
            # filter(lambda p: p.requires_grad, self.parameters()),
            self._model.parameters(),
            lr=effective_lr,
            betas=(args.beta1, args.beta2),
            weight_decay=args.weight_decay)
        return optimizer

    def _steps_per_epoch(self) -> float:
        dataset_size = len(self.train_loader)
        num_devices = max(1, self.trainer.num_devices)
        steps_per_epoch = math.ceil(
            dataset_size / num_devices / args.accumulate_grad)
        effective_batch_size = args.batch_size * num_devices * args.accumulate_grad
        effective_lr = args.lr * effective_batch_size / self._args.reference_batch_size
        print(f'Steps per Epoch: [{steps_per_epoch:_}], ',
              f'Effective Batch Size: [{effective_batch_size:_}], ',
              f'Effective LR: [{effective_lr:.2e}]')
        print('-'*80)
        return steps_per_epoch, effective_batch_size, effective_lr


if __name__ == '__main__':

    torch.set_float32_matmul_precision('medium')

    args = parse_finetune_args()

    train_loader, val_loader = build_data_loader(args)

    finetune_module = FinetuneModule(args, train_loader, val_loader)
    if args.compile:
        finetune_module = torch.compile(finetune_module)

    checkpoint_callback = ModelCheckpoint(
        save_top_k=3,
        monitor="val_acc1",
        mode="max",
        filename="{epoch:02d}-{val_acc1:.2f}-{val_loss:.2f}")

    trainer = L.Trainer(limit_train_batches=None,
                        max_epochs=args.epoch,
                        strategy=DDPStrategy(find_unused_parameters=True),
                        profiler="simple",
                        precision=args.precision,
                        accumulate_grad_batches=args.accumulate_grad,
                        gradient_clip_val=args.gradient_clipping,
                        log_every_n_steps=20,
                        callbacks=[
                            DeviceStatsMonitor(),
                            LearningRateMonitor(),
                            checkpoint_callback,
                        ])

    trainer.fit(model=finetune_module,
                train_dataloaders=train_loader,
                val_dataloaders=val_loader)
