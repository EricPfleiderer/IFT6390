import pickle
import os

import torch
import numpy as np
import pandas as pd

from src.transforms import *


class ElectricityConsumptionDataset(torch.utils.data.Dataset):

    def __init__(self, csv_file, root='data/'):
        self.root = root

        # Read the file
        self.df = pd.read_csv(root + csv_file)

        ids = self.df.iloc[:, 0]

        # Split sequences and convert to float
        sequences = self.df.iloc[:, 1].tolist()
        sequences = [torch.Tensor(np.array(sequence.split(' '), dtype=np.float)) for sequence in sequences]

        # Create new dataframe
        self.df = pd.DataFrame({'id': ids, 'sequence': sequences})

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        """
        :param idx: List (or tensor)
        :return:
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()
        return self.df.iloc[idx]


def sliding_windows(sequence, step_size=1, seq_length=500, batch_size=64, device=None):

    x = torch.empty(size=(0, seq_length), dtype=torch.float)
    y = torch.empty(size=(0,), dtype=torch.float)

    if device is not None:
        x = x.to(device)
        y = y.to(device)

    for i in range(0, len(sequence)-seq_length-1, step_size):

        # Build batch
        x = torch.cat((x, torch.unsqueeze(sequence[i:(i+seq_length)], dim=0)), dim=0)
        y = torch.cat((y, torch.unsqueeze(sequence[i+seq_length], dim=0)))

        # Only yield full batches
        if x.shape[0] >= batch_size:
            yield x, torch.unsqueeze(y, dim=1)

            x = torch.empty(size=(0, seq_length))
            y = torch.empty(size=(0,))

            if device is not None:
                x = x.to(device)
                y = y.to(device)


def get_processed_dataset(seq_len, step_size, transforms, shuffle=True, root='data/'):  # , horizon=1

    """
    Since the original dataset is composed of few very long sequences of variable length, we transform the dataset
    into sequences of fixed length by using a sliding window over the original data.

    :param seq_len:
    :param step_size:
    :param transforms:
    :param shuffle:
    :param root:
    :return:
    """

    # How to do this automatically? fetch dict of function params
    params = {
        'seq_len': seq_len,
        'step_size': step_size,
        'shuffle': shuffle,
        'transforms': transforms,
    }

    def dir_name():
        name = ""
        for key, value in params.items():
            name += f'{key}:{value}:'

        list_name = list(name)
        list_name[-1] = '/'
        return ''.join(list_name)

    # Load from existing
    if os.path.exists(root + dir_name()):
        print('Loading from existing save')
        train_x = pickle.load(open(root+dir_name()+'train_x.p', 'rb'))
        train_y = pickle.load(open(root+dir_name()+'train_y.p', 'rb'))

    # Generate from scratch and save
    else:
        print('Generating dataset from scratch.')

        # Create a new dir
        os.mkdir(root+dir_name())

        # Load the original dataset (preprocessed)
        base_train = ElectricityConsumptionDataset('train.csv')
        extracted_train = [base_train[i, 1] for i in range(len(base_train))]

        print('Applying transforms...')
        # Apply transforms through callables
        transformed_train = []
        for i in range(len(extracted_train)):
            transformed_sample, _ = apply_transforms(torch.unsqueeze(extracted_train[i], dim=0), params['transforms'])
            transformed_train.append(torch.squeeze(transformed_sample, dim=0))

        # Split the transformed sequences of variable length into subsequences of fixed length with sliding windows
        train_x = torch.empty((0, seq_len))
        train_y = torch.empty(0, 1)  # horizon
        for sequence in transformed_train:
            for batch_subsequences, batch_targets in sliding_windows(sequence, step_size, seq_len, 128):
                train_x = torch.cat((train_x, batch_subsequences), dim=0)
                train_y = torch.cat((train_y, batch_targets), dim=0)

        # Unsqueeze and shuffle
        if shuffle:
            train_x = train_x[torch.randperm(train_x.shape[0]), :]
        train_x = torch.unsqueeze(train_x, dim=-1)

        # Save the dataset if it doesn't yet exist
        pickle.dump(train_x, open(root+dir_name()+'train_x.p', 'wb'))
        pickle.dump(train_y, open(root+dir_name()+'train_y.p', 'wb'))
        pickle.dump(params, open(root+dir_name()+'params.p', 'wb'))

    print('Dataset fetched.')
    print('Dataset shape:', train_x.shape)

    return train_x, train_y, transforms

