import torch
from src.dataset import LogTransform, LDiff

dataset_space = {

    # Batch size affects computation time and the quality of the gradient estimate. Affects gpu memory.
    'batch_size': 1000,

    # Max value should be 720 as per our test set. Affects gpu memory.
    'seq_len': 500,

    # Controls the amount of data generated. Added data will be highly correlated. Does not affect GPU memory.
    'step_size': 50,

    # Transforms to be applied to the data. Order matters. Must be callable and non instantiated.
    'transforms': [],  # LogTransform, LDiff
}

LSTM_space = {

    # Training procedure params
    'num_epochs': 5,

    # Model params
    'model': {
        'num_layers': 6,  # affects gpu memory
        'hidden_size': 50,  # affects gpu memory
        'forecast_window': 120,
    },

    # Optimizer params
    'optimizer': {
        'type': torch.optim.Adam,
        'opt_params': {
            'lr': 0.0005,
            'weight_decay': 0.2
        }
    }
}