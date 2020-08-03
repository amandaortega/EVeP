from EVeP import EVeP
from math import sqrt
import numpy as np
from numpy import loadtxt
import optuna
import os
from sklearn.metrics import mean_squared_error
from tqdm import tqdm

class RMSE(object):
    def __init__(self, path, sigma_min, sigma_max, delta_min, delta_max, N_min, N_max, rho_min, rho_max, columns_ts):
        abspath = os.path.abspath(__file__)
        os.chdir(os.path.dirname(abspath))        

        try:
            self.X = loadtxt(path + 'X_train.csv', delimiter=',')
            self.y = loadtxt(path + 'Y_train.csv', delimiter=',')
        except Exception:
            self.X = loadtxt(path + 'X_train.csv', delimiter=' ')
            self.y = loadtxt(path + 'Y_train.csv', delimiter=' ')

        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.delta_min = delta_min
        self.delta_max = delta_max
        self.N_min = N_min
        self.N_max = N_max
        self.rho_min = rho_min
        self.rho_max = rho_max
        self.columns_ts = columns_ts

    def __call__(self, trial):
        sigma = trial.suggest_uniform('sigma', self.sigma_min, self.sigma_max)
        delta = trial.suggest_int('delta', self.delta_min, self.delta_max)
        N = trial.suggest_int('N', self.N_min, self.N_max)

        if self.rho_min is None or self.rho_max is None:
            rho = None
        else:
            rho = trial.suggest_loguniform('rho', self.rho_min, self.rho_max)

        model = EVeP(sigma, delta, N, rho, self.columns_ts)

        predictions = np.zeros((self.y.shape[0], 1))

        for i in tqdm(range(self.y.shape[0])):
            predictions[i, 0] = model.predict(self.X[i, :].reshape(1, -1))
            model.train(self.X[i, :].reshape(1, -1), self.y[i].reshape(1, -1), np.array([[i]]))
            
        # Desconsider the first prediction because there was no previous model 
        return sqrt(mean_squared_error(self.y[1:i+1], predictions[1:i+1]))    

sampler = optuna.samplers.TPESampler(seed=10)

# path = '../data/Mackey_Glass/'
# #study = optuna.create_study(sampler=sampler, study_name='Mackey_Glass_training', storage='sqlite:///mackey_glass.db', load_if_exists=True)
# study = optuna.create_study(sampler=sampler, study_name='Mackey_Glass_training_ls', storage='sqlite:///mackey_glass_ls.db', load_if_exists=True)
# #study.optimize(RMSE(path, 0, 0.3, 1, 50, 1, 5, 1e-5, 1e2), n_trials=100)
# study.optimize(RMSE(path, 0, 0.3, 1, 50, 1, 5, None, None), n_trials=100)

path = '../data/gas-furnace/'
#study = optuna.create_study(sampler=sampler, study_name='gas-furnace_training_ls', storage='sqlite:///gas-furnace_ls.db', load_if_exists=True)
study = optuna.create_study(sampler=sampler, study_name='gas-furnace_training', storage='sqlite:///gas-furnace.db', load_if_exists=True)
study.optimize(RMSE(path, 0, 0.5, 1, 50, 1, 5, 1e-5, 1e3), n_trials=100)