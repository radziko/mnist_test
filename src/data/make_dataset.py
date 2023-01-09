# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
import torch
import numpy as np
from torchvision import transforms
from torch.utils.data import TensorDataset


@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
def main(input_filepath, output_filepath):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')
    trains = [
    np.load(f"{input_filepath}/train_0.npz", mmap_mode="r"),
    np.load(f"{input_filepath}/train_1.npz", mmap_mode="r"),
    np.load(f"{input_filepath}/train_2.npz", mmap_mode="r"),
    np.load(f"{input_filepath}/train_3.npz", mmap_mode="r"),
    np.load(f"{input_filepath}/train_4.npz", mmap_mode="r")]

    images, labels = [], []
    for x in trains:
        images.append(x["images"])
        labels.append(x["labels"])

    train = [np.concatenate(images), np.concatenate(labels)]

    test_set = np.load(f"{input_filepath}/test.npz", mmap_mode="r")

    test = [test_set["images"], test_set["labels"]]
    
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5,), (0.5,)),
                                    transforms.Lambda(lambda x: torch.flatten(torch.swapdims(x, 0, 1), start_dim=1)),
                                    transforms.Lambda(lambda x: x.to(torch.float32))])

    trainset = {'images': transform(train[0]),
                'labels': train[1]}
    testset = {'images': transform(test[0]),
                'labels': test[1]}

    torch.save(trainset,f'{output_filepath}/_train.pt')
    torch.save(testset, f'{output_filepath}/_test.pt')


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()