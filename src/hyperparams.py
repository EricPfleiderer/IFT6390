import torch
from src.dataset import Scaler, LogTransform, LDiff

dataset_space = {
    'batch_size': 500,
    'seq_len': 700,  # MAX 720 FOR TEST SET
    'step_size': 50,  # Low step sizes generates more data (but the additional data is highly correlated)
    'transforms': [LogTransform, Scaler],  # , LDiff  # ORDER MATTERS! (sanity checks not yet implemented)
}

LSTM_space = {

    # Training procedure params
    'num_epochs': 10,

    # Model params
    'model': {
        'num_layers': 5,
        'hidden_size': 100,
    },

    # Optimizer params
    'optimizer': {
        'type': torch.optim.Adam,
        'opt_params': {
            'lr': 0.001,
        }
    }
}