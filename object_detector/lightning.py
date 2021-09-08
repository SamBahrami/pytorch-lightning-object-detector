import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torchvision.datasets import MNIST
from torchvision import transforms
from torchmetrics.functional import accuracy
import pytorch_lightning as pl


class ImageClassifier(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        # we use the maxpool multiple times, but define it once
        self.pool = nn.MaxPool2d(2, 2)
        # in_channels = 6 because self.conv1 output 6 channel
        self.conv2 = nn.Conv2d(6, 16, 5)
        # 5*5 comes from the dimension of the last convnet layer
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)  # no activation on final layer
        return x

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        z = self.forward(x)
        loss = F.cross_entropy(z, y)
        acc = accuracy(z, y)
        self.log("train/loss", loss)
        self.log("train/acc", acc)
        return loss

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        z = self.forward(x)
        loss = F.cross_entropy(z, y)
        acc = accuracy(z, y)
        self.log("val/loss", loss)
        self.log("val/acc", acc)


def main():
    # data
    dataset = MNIST("data", train=True, download=True, transform=transforms.ToTensor())
    mnist_train, mnist_val = random_split(dataset, [55000, 5000])

    train_loader = DataLoader(mnist_train, batch_size=256, num_workers=6)
    val_loader = DataLoader(mnist_val, batch_size=256, num_workers=6)

    # model
    model = ImageClassifier()

    # training
    trainer = pl.Trainer(gpus=1, max_epochs=20)
    trainer.fit(model, train_loader, val_loader)


if __name__ == "__main__":
    main()
