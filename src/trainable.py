from typing import Union

import numpy as np
import torch
import matplotlib.pyplot as plt

from src.dataset import get_processed_dataset
from src.models import get_model_by_name
from src.transforms import *


class TorchTrainable:

    """
    Wrapper for torch models, their optimizer and their training procedure. Also saves training history for plotting.
    """

    def __init__(self, params):

        # Parameters
        self.params = params
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.transforms = params['transforms']

        # Processed data set
        self.train_x, self.train_y, self.val_x, self.val_y, self.scaler = get_processed_dataset(
            self.params['seq_len'], self.params['step_size'], self.params['transforms'],
            self.params['model']['forecast_window'], self.params['train_val_split'])

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

        # Model performance is evaluated in the transformed feature space
        for epoch in range(self.params['num_epochs']):

            # Training
            avg_loss = 0
            for i in range(len(self.train_x)//self.params['batch_size']):

                # Batch and predictions
                idx_start = i * self.params['batch_size']
                idx_end = (i+1) * self.params['batch_size']
                sequences = self.train_x[idx_start:idx_end, :].to(self.model.device)
                targets = self.train_y[idx_start:idx_end, :].to(self.model.device)
                predictions = self.infer(torch.unsqueeze(sequences, dim=2))

                if epoch == 2:
                    a = 10
                self.optimizer.zero_grad()
                batch_loss = self.criterion(predictions, targets)
                batch_loss.backward()
                avg_loss += batch_loss.item()
                self.optimizer.step()

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
                    predictions = self.infer(torch.unsqueeze(sequences, dim=2))

                    # Mean val loss through batch
                    batch_loss = self.criterion(predictions, targets)
                    avg_loss += batch_loss.item()

            # Average validation loss
            avg_loss /= len(self.val_x)//self.params['batch_size']
            self.history['val_loss'].append(avg_loss)

            print(f"epoch #{epoch}: train loss: %1.8f, val loss: %1.8f"
                  % (self.history['train_loss'][-1], self.history['val_loss'][-1]))

    def infer(self, x: Union[np.array, torch.Tensor]):
        """
         x is assumed to be processed and transformed. For internal use.
        :param x:
        :return:
        """
        return self.model(x.to(self.device))

    def __call__(self,  x: Union[np.array, torch.Tensor]):

        """
        Predicts the next time step given a sequence or batch of sequences.
        x is assumed to be unprocessed and untransformed.

        :param x: Can be a 1 or 2 dimension np array or torch tensor.
        If x has 1 dimension, we assume it is a sequence.
        If x has 2 dimensions, we assume it is a batch of sequence.

        :return: Batch of model predictions.
        """

        # Cast to torch tensor if input is np array
        if type(x) == np.array:
            x = torch.tensor(x)

        # If x has a single dimension, we assume it is a simple sequence. We batch it.
        if len(x.shape) == 1:
            x = torch.unsqueeze(x, dim=0)

        from src.dataset import scale_batch_row_vectors, rescale_batch_row_vectors

        scaled_x = scale_batch_row_vectors(x.cpu().detach(), self.scaler)

        # We unsqueeze the number of features (1 in our case).
        input_x = torch.unsqueeze(scaled_x, dim=-1).to(self.device)

        predictions = self.infer(input_x)

        rescaled_predictions = rescale_batch_row_vectors(predictions.cpu().detach(), self.scaler)

        return rescaled_predictions.cpu().detach()

    def plot_history(self, path):

        plt.figure()
        plt.plot(range(len(self.history['train_loss'])), self.history['train_loss'], label='training')
        plt.plot(range(len(self.history['val_loss'])), self.history['val_loss'], label='validation')
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.legend()
        plt.savefig(path+'Training_loss.png')

