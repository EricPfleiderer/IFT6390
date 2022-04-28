import torch
from src.dataset import Scaler, LogTransform, LDiff

dataset_space = {

    # Batch size affects computation time and the quality of the gradient estimate. Affects gpu memory.
    'batch_size': 512,

    # Max value should be 720 as per our test set. Affects gpu memory.
    'seq_len': 240,

    # Controls the amount of data generated. Added data will be highly correlated. Does not affect GPU memory.
    'step_size': 10,

    # Transforms to be applied to the data. Order matters. Must be callable and non instantiated.
    # TODO: implement l-differencing transform
    'transforms': [LogTransform, Scaler],  # , LDiff
}

LSTM_space = {

    # Training procedure params
    'num_epochs': 5,

    # Model params
    'model': {
        'num_layers': 5,  # affects gpu memory
        'hidden_size': 50,  # affects gpu memory
    },

    # Optimizer params
    'optimizer': {
        'type': torch.optim.Adam,
        'opt_params': {
            'lr': 0.001,
            'weight_decay': 0.1
        }
    }
}