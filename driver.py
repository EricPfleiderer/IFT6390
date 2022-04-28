import matplotlib.pyplot as plt
import torch

from src.dataset import ElectricityConsumptionDataset
from src.trainable import TorchTrainable
from src.hyperparams import *
from src.config import configs
from datetime import datetime
import os
import pickle

# TODO: implement l-differencing
# TODO: revise / debug / sanity checks
# TODO: hyperparam optimization
# TODO: bagging


def run_experiment(from_save_path=None, root='experiments/'):

    """
    Runs an experiment either from scratch or from save. The experiment consists of training a model and
    then forcasting the next time steps. Training graphs, the model and its hyperparameters are stored in a new dir.

    :param from_save_path: Path towards a previous experiment
    :param root: root directory for archiving of individual experiments
    :return:
    """

    # Creating a dir for the experiment
    now = datetime.now()
    full_path = root + 'Experiment_' + str(now) + '/'
    os.makedirs(full_path)

    # Load model from previous experiment
    if from_save_path is not None:
        print('Loading model from previous experiment')
        trainable = pickle.load(open(from_save_path+'trainable.p', 'rb'))
        params = pickle.load(open(from_save_path+'params.p', 'rb'))

    # Train a new model
    else:
        print('Training new model')
        params = dict(**dataset_space, **LSTM_space, **configs)
        trainable = TorchTrainable(params)
        trainable.train()

    trainable.plot_history(full_path)

    pickle.dump(trainable, open(full_path+'trainable.p', 'wb'))
    pickle.dump(params, open(full_path+'params.p', 'wb'))

    test_set = ElectricityConsumptionDataset('test.csv')

    test_idx = 0
    test_sequence = test_set[test_idx, 1][0:params['seq_len']]

    forecast_window = 20
    forecast = torch.empty((0,))

    # Autoregressive forecasting
    for i in range(forecast_window):
        current_seq = torch.cat((test_sequence[i:], forecast))
        next_value = torch.squeeze(trainable(current_seq), dim=1)
        forecast = torch.cat((next_value, forecast))

    full_y = test_set[test_idx, 1][0:params['seq_len']+forecast_window]
    y_pred = forecast

    # Plot and save predictions
    plt.figure()
    plt.plot(range(params['seq_len'] - forecast_window,
                   params['seq_len'] + forecast_window),

             full_y[params['seq_len'] - forecast_window:
                    params['seq_len'] + forecast_window], label='true sequence')

    plt.plot(range(params['seq_len'], params['seq_len'] + forecast_window), y_pred, label='forecast', linestyle='--')
    plt.legend()
    plt.savefig(full_path+'prediction.png')


run_experiment()
# run_experiment(from_save_path='experiments/Experiment_2022-04-28 01:13:46.474846')

