import torch

dataset_space = {
    'batch_size': 512,
    'seq_len': 250,
    'step_size': 20,
    # TODO: add transforms
}

LSTM_space = {

    # Training procedure params
    'num_epochs': 20,

    # Model params
    'model': {
        'num_layers': 10,
        'hidden_size': 150,
    },

    # Optimizer params
    'optimizer': {
        'type': torch.optim.Adam,
        'opt_params': {
            'lr': 0.001,
        }
    }
}