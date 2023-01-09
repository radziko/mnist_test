import argparse
import sys

import torch
import click
from torch import nn
from model import MyAwesomeModel
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
@click.argument("model_checkpoint")
def evaluate(model_checkpoint):
    print("Evaluating until hitting the ceiling")
    print(model_checkpoint)

    # TODO: Implement evaluation logic here
    state_dict = torch.load(model_checkpoint)
    model = MyAwesomeModel()
    model.load_state_dict(state_dict)
    test_set = torch.load('data/processed/_test.pt')

    testset = TensorDataset(torch.Tensor(test_set['images']), torch.Tensor(test_set['labels']))

    # Download and load the training data
    testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=True)

    model.eval()
    accuracy = 0
    with torch.no_grad():
        for images, labels in testloader:
            probs, _ = model.forward(images)

            top_p, top_class = probs.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)

            accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

    accuracy = accuracy / len(testloader)
    print(f'Accuracy: {accuracy * 100}%')


cli.add_command(evaluate)


if __name__ == "__main__":
    cli()
    