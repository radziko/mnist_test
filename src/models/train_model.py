import argparse
import sys

import torch
import click
from torch import nn
from src.models.model import MyAwesomeModel
from torchvision import transforms
from torch.utils.data import TensorDataset
from torch.optim import Adam
import matplotlib.pyplot as plt
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


@click.group()
def cli():
    pass


@click.command()
@click.option("--lr", default=1e-3, help='learning rate to use for training')
def train(lr):
    print("Training day and night")
    print(lr)

    # TODO: Implement training loop here
    model = MyAwesomeModel()
    train_set = torch.load('data/processed/_train.pt')

    trainset = TensorDataset(torch.Tensor(train_set['images']), torch.Tensor(train_set['labels']))

    # Download and load the training data
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

    criterion = nn.NLLLoss()
    optimizer = Adam(model.parameters(), lr=lr)

    epochs = 30
    train_loss = []
    for e in range(epochs):
        running_loss = 0
        print(f"Begin epoch: #{e}")
        for images, labels in trainloader:
            optimizer.zero_grad()

            log_ps, _  = model(images)
            loss = criterion(log_ps, labels.to(torch.long))
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Loss={running_loss}")
        train_loss.append(running_loss)
    plt.figure()
    plt.plot(train_loss)
    plt.savefig('reports/figures/training_loss.png')

    torch.save(model.state_dict(), 'models/train_checkpoint.pth')



cli.add_command(train)


if __name__ == "__main__":
    cli()