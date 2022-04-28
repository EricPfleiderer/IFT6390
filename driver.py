import matplotlib.pyplot as plt

from src.dataset import ElectricityConsumptionDataset
from src.trainable import TorchTrainable
from src.hyperparams import *
from src.config import configs
from datetime import datetime
import os
import pickle

# TODO: revise loss (make scale independant of params)
# TODO: hyperparam optimization
# TODO: bagging


def plot_seq(x):

    plt.Figure()
    plt.plot(range(len(x)),x)
    plt.show()


def run_experiment(root='experiments/'):

    now = datetime.now()
    full_path = root + 'Experiment_' + str(now) + '/'
    os.makedirs(full_path)

    params = dict(**dataset_space, **LSTM_space, **configs)

    trainable = TorchTrainable(params)

    trainable.train()
    trainable.plot_history(full_path)

    pickle.dump(trainable, open(full_path+'trainable.p', 'wb'))
    pickle.dump(params, open(full_path+'params.p', 'wb'))

    test_set = ElectricityConsumptionDataset('test.csv')

    x=10


run_experiment()

