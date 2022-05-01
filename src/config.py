import torch

configs = {
    'model_name': 'lstm',
    'train_val_split': 0.85,
    'criterion': torch.nn.MSELoss(),
}