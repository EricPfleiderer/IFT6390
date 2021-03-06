import matplotlib.pyplot as plt
import pandas as pd
import torch

from src.dataset import ElectricityConsumptionDataset
from src.trainable import TorchTrainable
from src.hyperparams import *
from src.config import configs
from datetime import datetime
import os
import pickle


# TODO: debug model (always predicts same thing, learning the avg??) PRIORITY!!!!
# TODO: debug transforms
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

    out = torch.empty((0, 120))

    print('Forecasting...')
    for i in range(len(test_set)):
        if i % 100 == 0:
            print(i)
        forecast = torch.squeeze(trainable(test_set[i, 1][-params['seq_len']:]))
        out = torch.cat((out, torch.unsqueeze(forecast, dim=0)), dim=0)

    cols = [f'Prediction_{i}' for i in range(1, 121)]

    out_df = pd.DataFrame(out, columns=cols)

    out_df.to_csv(full_path+'submission.csv', index_label=True)

    # # Test samples
    # test_idxs = (1, 2, 3, 5, 10, 20, 50)
    # test_sequences = torch.empty((0, params['seq_len']))
    # for idx in test_idxs:
    #     test_sequences = torch.cat((test_sequences, torch.unsqueeze(test_set[idx, 1][0:params['seq_len']], dim=0)))
    #
    #
    # # Test samples extended to cover the prediction horizon
    # full_sequences = torch.empty((0, params['seq_len']+params['model']['forecast_window']))
    # for idx in test_idxs:
    #     full_sequences = torch.cat((full_sequences, torch.unsqueeze(test_set[idx, 1][0:params['seq_len']+params['model']['forecast_window']], dim=0)))
    #
    # # Model predictions
    # forecasts = torch.squeeze(trainable(test_sequences), dim=0)
    #
    # # Print a few samples
    # for forecast, full_y, idx in zip(forecasts, full_sequences, test_idxs):
    #
    #     # Plot and save predictions
    #     plt.figure()
    #     plt.plot(range(params['seq_len'] - params['model']['forecast_window'],
    #                    params['seq_len'] + params['model']['forecast_window']),
    #
    #              full_y[params['seq_len'] - params['model']['forecast_window']:
    #                     params['seq_len'] + params['model']['forecast_window']], label='true sequence')
    #
    #     plt.plot(range(params['seq_len'], params['seq_len'] + params['model']['forecast_window']), forecast, label='forecast', linestyle='--')
    #     plt.legend()
    #     plt.savefig(full_path+f'prediction_idx{idx}.png')


# run_experiment('experiments/Experiment_2022-04-28 23:44:02.871440/')

run_experiment()
