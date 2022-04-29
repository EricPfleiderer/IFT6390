import torch

configs = {
    'model_name': 'lstm',
    'train_val_split': 0.9,
    'criterion': torch.nn.MSELoss(),
}