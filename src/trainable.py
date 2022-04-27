from typing import Union

import numpy as np
import torch
import matplotlib.pyplot as plt

from src.dataset import get_processed_dataset
from src.models import get_model_by_name


class TorchTrainable:

    """
    Wrapper for torch models, their optimizer and their training procedure. Also saves training history for plotting.
    """

    def __init__(self, params):

        # Parameters
        self.params = params
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Processed data set
        self.x, self.y = get_processed_dataset(self.params['seq_len'], self.params['step_size'])

        # Split into train and validation set
        split_idx = int(params['train_val_split'] * self.x.shape[0])
        self.train_x, self.train_y = self.x[:split_idx, :], self.y[:split_idx, :]
        self.val_x, self.val_y = self.x[split_idx:, :], self.y[split_idx:, :]

        # Torch model
        self.model = get_model_by_name(params['model_name'])
        self.model = self.model.to(self.device)

        # Torch optimizer
        self.optimizer = params['optimizer']['type'](self.model.parameters(), **params['optimizer']['opt_params'])
        self.criterion = params['criterion']

        # Training statistics
        self.history = {
            'train_loss': [],
            'val_loss': [],
        }

    def reset_history(self):
        self.history = {
            'train_loss': [],
            'val_loss': [],
        }

    def train(self):

        for epoch in range(self.params['num_epochs']):

            # Training
            avg_loss = 0
            for i in range(len(self.train_x)//self.params['batch_size']):

                # Batch and predictions
                idx_start = i * self.params['batch_size']
                idx_end = (i+1) * self.params['batch_size']
                sequences = self.train_x[idx_start:idx_end, :].to(self.model.device)
                targets = self.train_y[idx_start:idx_end, :].to(self.model.device)
                predictions = self.model(sequences)

                # Gradient update
                self.optimizer.zero_grad()
                batch_loss = self.criterion(predictions, targets)
                batch_loss.backward()
                self.optimizer.step()

                avg_loss += batch_loss.item()

            # Average training loss
            avg_loss /= len(self.train_x)//self.params['batch_size']
            self.history['train_loss'].append(avg_loss)

            # Validation
            avg_loss = 0
            for i in range(len(self.val_x)//self.params['batch_size']):

                with torch.no_grad():
                    # Batch and predictions
                    idx_start = i * self.params['batch_size']
                    idx_end = (i+1) * self.params['batch_size']
                    sequences = self.val_x[idx_start:idx_end, :].to(self.model.device)
                    targets = self.val_y[idx_start:idx_end, :].to(self.model.device)
                    predictions = self.model(sequences)

                    # Mean val loss through batch
                    batch_loss = self.criterion(predictions, targets)
                    avg_loss += batch_loss.item()

            # Average validation loss
            avg_loss /= len(self.val_x)//self.params['batch_size']
            self.history['val_loss'].append(avg_loss)

            print(f"epoch #{epoch}: train loss: %1.0f, val loss: %1.0f"
                  % (self.history['train_loss'][-1], self.history['val_loss'][-1]))

    def infer(self, x: Union[np.array, torch.Tensor]):

        """
        Forward pass through the model.

        :param x: Can be a 1, 2 or 3 dimension np array or torch tensor.
        If x has 1 dimension, we assume it is a sequence.
        If x has 2 dimensions, we assume it is a batch of sequence.
        If x has 3 dimensions, we assume it is a batch of sequences with a third dim for features.

        :return: Batch of model predictions.
        """

        # Cast to torch tensor if input is np array
        if type(x) == np.array:
            x = torch.tensor(x)

        # If x has a single dimension, we assume it is a simple sequence. We batch it.
        if len(x.shape) == 1:
            x = torch.unsqueeze(x, dim=0)

        # If x has 2 dimensions, we assume it is a batch of sequences.
        # We unsqueeze the number of features (1 in our case).
        if len(x.shape) == 2:
            x = torch.unsqueeze(x, dim=-1)

        if len(x.shape) > 3:
            raise ValueError(f'Expected data shape is of length 1, 2 or 3. Current length is {len(x.shape)}.')

        return self.model(x.to(self.device))

    def __call__(self, x):
        return self.infer(x)

    def plot_history(self, path):

        plt.figure()
        plt.plot(range(len(self.history['train_loss'])), self.history['train_loss'], label='training')
        plt.plot(range(len(self.history['val_loss'])), self.history['val_loss'], label='validation')
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.legend()
        plt.savefig(path+'Training_loss.png')

