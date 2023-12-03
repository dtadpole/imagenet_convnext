import os
import argparse
from torch import optim, nn, utils, Tensor
from torchvision import datasets, transforms
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
from model import ConvNeXt

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('-f', '--folder', default='./imagenet/',
                    help='path to dataset (default: ./imagenet/)')
parser.add_argument('--batch_size', default=96, type=int,
                    help="batch size")
parser.add_argument('--workers', default=5, type=int,
                    help="number of workers")
parser.add_argument('--numops', default=2, type=int,
                    help='number of ops, default 2')
parser.add_argument('--magnitude', default=15, type=int,
                    help="magnitude (default: 15)")
args = parser.parse_args()

train_dataset = datasets.ImageFolder(
    os.path.join(args.folder, 'train'),
    transforms.Compose([
        transforms.RandAugment(num_ops=args.numops, magnitude=args.magnitude),
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
    train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, prefetch_factor=10)
val_loader = utils.data.DataLoader(
    val_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, prefetch_factor=10)


class Model(L.LightningModule):
    def __init__(self):
        super().__init__()
        self._model = ConvNeXt(depths=[3, 3, 9, 3], dims=[96, 192, 384, 768])
        self._loss = nn.CrossEntropyLoss()
        self.save_hyperparameters()

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        # it is independent of forward
        images, target = batch
        output = self._model(images)
        loss = self._loss(output, target)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        images, target = batch
        output = self._model(images)
        loss = self._loss(output, target)
        self.log("val_loss", loss)
        # measure accuracy and record loss
        # acc1, acc5 = accuracy(output, target, topk=(1, 5))
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        return optimizer


model = Model()

checkpoint_callback = ModelCheckpoint(
    save_top_k=3, monitor="val_loss", filename="model-{epoch:02d}-{val_loss:.2f}")

trainer = L.Trainer(limit_train_batches=500, max_epochs=5,
                    callbacks=[checkpoint_callback])
trainer.fit(model=model, train_dataloaders=train_loader,
            val_dataloaders=val_loader)
