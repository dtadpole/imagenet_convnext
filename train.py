import os
import math
import argparse
from copy import deepcopy
import wandb
import torch
from torch import optim, nn, utils, Tensor
from torch.utils.data import DataLoader
import torch.distributed as dist
from torchvision import datasets, transforms
from torchvision.transforms.functional import InterpolationMode
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
import lightning as L
from lightning.pytorch.strategies import DDPStrategy
from lightning.pytorch.callbacks import ModelCheckpoint, DeviceStatsMonitor, LearningRateMonitor, GradientAccumulationScheduler
from convnext import convnext_tiny, convnext_small, convnext_small_2, convnext_base, convnext_large, convnext_xlarge
from maxvit import max_vit_tiny_224, max_vit_small_224, max_vit_base_224, max_vit_large_224
from maxvit import MaxViT
from timm.data.mixup import Mixup
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from flops_profiler.profiler import get_model_profile

wandb_project = "ImageNet"


def parse_pretrain_args():
    # basic params
    parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
    parser.add_argument('-f', '--folder', default='./imagenet/',
                        help='path to dataset (default: ./imagenet/)')
    parser.add_argument('-a', '--arch', default='ConvNeXt_S',
                        help='model arch (default: ConvNeXt_S)')
    parser.add_argument('-b', '--batch_size', default=64, type=int,
                        help="batch size (default: 64)")
    parser.add_argument('-r', '--resume', default=None, type=str,
                        help="resume checkpoint path (default: None)")

    # epoch and lr
    parser.add_argument('--epoch', default=90, type=int,
                        help="total epoch (default: 90)")
    parser.add_argument('--warmup_epoch', default=10, type=float,
                        help='warmup epoch (default: 10)')
    parser.add_argument('--lr', default=3e-4, type=float,
                        help="learning rate (default: 3e-4)")
    parser.add_argument('--lr_end', default=1e-6, type=float,
                        help="ending learning rate (default: 1e-6)")
    parser.add_argument('--accumulate_grad', default=8, type=int,
                        help="accumulate gradient (default: 8)")
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
    parser.add_argument('--weight_decay', default=0.1, type=float,
                        help='weight decay (default: 0.1)')
    parser.add_argument('--model_ema_decay', default=0.995, type=float,
                        help='model ema decay (default: 0.995)')

    # workers
    parser.add_argument('--compile', default=False, type=bool,
                        help="compile model (default: False)")
    parser.add_argument('--precision', default='bf16-mixed', type=str,
                        help='training precision (default: bf16-mixed)')
    parser.add_argument('--workers', default=10, type=int,
                        help="number of workers (default: 10)")
    parser.add_argument('--prefetch', default=5, type=int,
                        help="number of prefetch (default: 5)")

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

    # transformds
    # parser.add_argument('--transform_ops', default=2, type=int,
    #                     help='number of ops, default 2')
    # parser.add_argument('--transform_mag', default=15, type=int,
    #                     help="magnitude (default: 15)")
    parser.add_argument('--random_erase', default=0.1, type=float,
                        help="random erase (default: 0.1)")
    parser.add_argument('--train_crop_size', default=192, type=int,
                        help='train crop size (default 192)')
    args = parser.parse_args()
    return args


def build_model(args):
    if args.arch.lower() == "ConvNeXt_T".lower():
        return convnext_tiny(drop_path_rate=args.drop_path_rate)
    elif args.arch.lower() == "ConvNeXt_S".lower():
        return convnext_small(drop_path_rate=args.drop_path_rate)
    elif args.arch.lower() == "ConvNeXt_S2".lower():
        return convnext_small_2(drop_path_rate=args.drop_path_rate)
    elif args.arch.lower() == "ConvNeXt_B".lower():
        return convnext_base(drop_path_rate=args.drop_path_rate)
    elif args.arch.lower() == "ConvNeXt_L".lower():
        return convnext_large(drop_path_rate=args.drop_path_rate)
    elif args.arch.lower() == "ConvNeXt_XL".lower():
        return convnext_xlarge(drop_path_rate=args.drop_path_rate)
    elif args.arch.lower() == "MaxViT_T".lower():
        return max_vit_tiny_224(drop=args.drop_rate, attn_drop=args.drop_rate, drop_path=args.drop_path_rate)
    elif args.arch.lower() == "MaxViT_S".lower():
        return max_vit_small_224(drop=args.drop_rate, attn_drop=args.drop_rate, drop_path=args.drop_path_rate)
    elif args.arch.lower() == "MaxViT_B".lower():
        return max_vit_base_224(drop=args.drop_rate, attn_drop=args.drop_rate, drop_path=args.drop_path_rate)
    elif args.arch.lower() == "MaxViT_L".lower():
        return max_vit_large_224(drop=args.drop_rate, attn_drop=args.drop_rate, drop_path=args.drop_path_rate)
    else:
        raise Exception('Unknown arch %s' % args.arch)


def build_data_loader(args):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])

    # train dataset
    train_dataset = datasets.ImageFolder(
        os.path.join(args.folder, 'train'),
        transforms.Compose([
            # transforms.RandAugment(num_ops=args.transform_ops,
            #                        magnitude=args.transform_mag),
            # transforms.Resize(256, interpolation=InterpolationMode.BILINEAR),
            # transforms.CenterCrop(224),
            transforms.RandomResizedCrop(args.train_crop_size, interpolation=InterpolationMode.BILINEAR, antialias=True),
            transforms.TrivialAugmentWide(interpolation=InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.RandomErasing(p=args.random_erase),
            normalize,
        ]))

    val_dataset = datasets.ImageFolder(
        os.path.join(args.folder, 'val'),
        transforms.Compose([
            transforms.Resize(232, interpolation=InterpolationMode.BILINEAR, antialias=True),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
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

# mixup


def build_mixup_fn(args):
    mixup_fn = None
    mixup_active = args.mixup > 0 or args.cutmix > 0. or args.cutmix_minmax is not None
    if mixup_active:
        print("Mixup is activated!")
        mixup_fn = Mixup(
            mixup_alpha=args.mixup, cutmix_alpha=args.cutmix, cutmix_minmax=args.cutmix_minmax,
            prob=args.mixup_prob, switch_prob=args.mixup_switch_prob, mode=args.mixup_mode,
            label_smoothing=args.smoothing)
    return mixup_fn


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

class EMA(nn.Module):
    """ Model Exponential Moving Average V2 from timm"""

    def __init__(self, model, decay=0.9999):
        super(EMA, self).__init__()
        # make a copy of the model for accumulating moving average of weights
        self.module = deepcopy(model)
        self.module.eval()
        self.decay = decay

    def _update(self, model, update_fn):
        with torch.no_grad():
            for ema_v, model_v in zip(self.module.state_dict().values(), model.state_dict().values()):
                ema_v.copy_(update_fn(ema_v, model_v))

    def update(self, model):
        self._update(model, update_fn=lambda e,
                     m: self.decay * e + (1. - self.decay) * m)

    def set(self, model):
        self._update(model, update_fn=lambda e, m: m)

class PreTrainModule(L.LightningModule):

    def __init__(self, args, train_loader, val_loader):
        super().__init__()
        self._args = args
        self.train_loader = train_loader
        self.val_loader = val_loader
        self._model = build_model(args)
        self._model_ema = EMA(self._model, decay=self._args.model_ema_decay)
        self._mixup_fn = build_mixup_fn(args)
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
        self.train_step_outputs = []
        self.validation_step_outputs = []
        self.wandb_inited = False
        self._profiled = False

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop
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
        sch = self.lr_schedulers()
        lr = sch.get_last_lr()[0]
        # log
        self.log_dict({
            "train_loss": loss,
            "train_acc1": acc1,
            "train_acc5": acc5,
            "train_lr": lr,
        })
        # buffer
        self.train_step_outputs.append({
            "train_loss": loss,
            "train_acc1": acc1,
            "train_acc5": acc5,
            "train_lr": lr,
        })
        # return
        return loss

    def optimizer_step(
        self,
        epoch,
        batch_idx,
        optimizer,
        optimizer_closure,
    ):
        # default lightning module
        optimizer.step(closure=optimizer_closure)
        # update ema model
        self._model_ema.update(self._model)
        # step lr scheduler
        sch = self.lr_schedulers()
        sch.step()

    def on_train_epoch_end(self):
        # keeps previous epoch info
        print(" ", end="")

    def validation_step(self, batch, batch_idx):
        images, targets = batch
        if self.trainer.local_rank == 0:
            if not self._profiled:
                flops, macs, params = get_model_profile(
                    self._model,
                    input_shape=tuple(images.shape),
                    args=[images],
                    print_profile=True,
                    detailed=False,
                    as_string=True,
                )
                self._profiled = True
                print(f'FLOPS: {flops}, MACS: {macs}, PARAMS: {params}')
                print('-'*80)
                # print steps, batch size and LR
                self._steps_per_epoch()
        # validation_step defines the validation loop.
        output = self._model_ema.module(images)
        loss = self._eval_loss_fn(output, targets)
        acc1, acc5 = accuracy(output, targets, topk=(1, 5))
        log_dict = {
            "val_loss": loss,
            "val_acc1": acc1,
            "val_acc5": acc5,
        }
        self.validation_step_outputs.append(log_dict)
        self.log_dict(log_dict, sync_dist=True, rank_zero_only=False)

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
        # return if sanity checking
        if self.trainer.sanity_checking:
            return
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
                wandb.init(project=wandb_project, name=wandb_name,
                           group="PreTrain", config=self._args)
                self.wandb_inited = True
            wandb.log(log_dict)

    def configure_optimizers(self):
        steps_per_epoch, _, effective_lr, effective_lr_end = self._steps_per_epoch()
        optimizer = optim.AdamW(
            # self.parameters(),
            # filter(lambda p: p.requires_grad, self.parameters()),
            self._model.parameters(),
            lr=effective_lr,
            betas=(args.beta1, args.beta2),
            weight_decay=args.weight_decay)
        warmup_scheduler = LinearLR(optimizer,
                                    start_factor=1e-4,
                                    end_factor=1.0,
                                    total_iters=steps_per_epoch*args.warmup_epoch)
        main_scheduler = CosineAnnealingLR(optimizer,
                                           steps_per_epoch *
                                           (args.epoch-args.warmup_epoch),
                                           eta_min=effective_lr_end)
        scheduler = SequentialLR(optimizer,
                                 schedulers=[
                                     warmup_scheduler,
                                     main_scheduler],
                                 milestones=[steps_per_epoch * args.warmup_epoch])
        return [optimizer], [scheduler]

    def _steps_per_epoch(self) -> float:
        dataset_size = len(self.train_loader)
        num_devices = max(1, self.trainer.num_devices)
        steps_per_epoch = math.ceil(
            dataset_size / num_devices / args.accumulate_grad)
        effective_batch_size = args.batch_size * num_devices * args.accumulate_grad
        effective_lr = args.lr * effective_batch_size / self._args.reference_batch_size
        effective_lr_end = args.lr_end * effective_batch_size / \
            self._args.reference_batch_size
        print(f'Steps per Epoch: [{steps_per_epoch:_}], ',
              f'Effective Batch Size: [{effective_batch_size:_}], ',
              f'Effective LR: [{effective_lr:.2e}, {effective_lr_end:.2e}]')
        print('-'*80)
        return steps_per_epoch, effective_batch_size, effective_lr, effective_lr_end


if __name__ == '__main__':

    torch.set_float32_matmul_precision('medium')

    args = parse_pretrain_args()

    train_loader, val_loader = build_data_loader(args)

    pretrain_module = PreTrainModule(args, train_loader, val_loader)
    if args.compile:
        print('Compiling...')
        pretrain_module = torch.compile(pretrain_module)
        print('Compiled.')

    checkpoint_callback = ModelCheckpoint(
        save_top_k=3,
        monitor="val_acc1",
        mode="max",
        filename="{epoch:02d}-{val_acc1:.2f}-{val_loss:.2f}")

    trainer = L.Trainer(limit_train_batches=None,
                        max_epochs=args.epoch,
                        strategy = DDPStrategy(find_unused_parameters=True),
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

    trainer.fit(model=pretrain_module,
                train_dataloaders=train_loader,
                val_dataloaders=val_loader,
                ckpt_path=args.resume)
