import torch
from src.dataset import Scaler, LogTransform, LDiff

dataset_space = {
    'batch_size': 1000, # affect gpu memory
    'seq_len': 200,  # MAX 720 FOR TEST SET, affects gpu memory
    'step_size': 100,  # Low step sizes generates more data, does not affect GPU memory
    'transforms': [LogTransform, Scaler],  # , LDiff  # ORDER MATTERS! (sanity checks not yet implemented)
}

LSTM_space = {

    # Training procedure params
    'num_epochs': 30,

    # Model params
    'model': {
        'num_layers': 10,  # affects gpu memory
        'hidden_size': 100, # affects gpu memory
    },

    # Optimizer params
    'optimizer': {
        'type': torch.optim.Adam,
        'opt_params': {
            'lr': 0.001,
            'weight_decay': 0.05
        }
    }
}