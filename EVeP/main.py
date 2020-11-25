import matplotlib
matplotlib.use('Agg')
import csv
from EVeP import EVeP
from math import sqrt
import matplotlib.pyplot as plt
import mlflow
import numpy as np
from numpy import loadtxt
import os
from sklearn.metrics import mean_squared_error
import time
from tqdm import tqdm
import warnings

# databases IDs
PLANT_IDENTIFICATION = 1
MACKEY_GLASS = 2
SP_500 = 3
TEMPERATURE = 4
WIND = 5
RAIN = 6
GAS_FURNACE = 7
SYSTEM_IDENTIFICATION_2 = 8
AUTO_MPG = 9
AUTOS = 10
AILERONS = 11
TRIAZINES = 12
HELICOPTER = 13
QUADCOPTER = 14

# algorithm versions
LS = 0
MTL = 1

VERSION_NAME = ['LS', 'MTL']

TRAINING = 0
TEST = 1

def read_csv(file):
    try:
        return loadtxt(file)
    except Exception:
        return loadtxt(file, delimiter=',')

def plot_graph(y, y_label, x_label, file_name, y_aux=None, legend=None, legend_aux=None):
    plt.plot(y)
    plt.ylabel(y_label)
    plt.xlabel(x_label)    

    if y_aux is not None:    
        plt.plot(y_aux)
        plt.legend([legend, legend_aux])
    else:
        plt.annotate(str(round(y[-1, 0], 3)), xy=(y.shape[0], y[-1, 0]), ha='center')
    
    plt.savefig(file_name)
    plt.close()

def read_parameters():
    try:
        algorithm = int(input('Enter the version of EVeP to run:\n1- LS\n2- MTL (default)\n')) - 1
    except ValueError:
        algorithm = MTL

    try:
        mode = int(input('Run the 1- training or 2- test (default) dataset?\n')) - 1
    except ValueError:
        mode = TEST  

    try:
        dataset = int(input('Enter the dataset to be tested:\n1- Nonlinear Dynamic Plant Identification With Time-Varying Characteristics (default)\n' + 
        '2- Mackey–Glass Chaotic Time Series (Long-Term Prediction)\n3- Online Prediction of S&P 500 Daily Closing Price\n' + 
        '4- Wheater temperature\n5- Wind speed\n6- Rain\n7- Gas furnace\n8- Nonlinear System Identification 2\n9- Auto-MPG\n10- Autos\n11- Ailerons\n12- Triazines\n13- Helicopter\n14- Quadcopter\n'))
    except ValueError:
        dataset = PLANT_IDENTIFICATION

    dim = -1
    N_default = 4
    columns_ts = None

    if dataset == PLANT_IDENTIFICATION:
        sites = ['Default']
        input_path_default = '../data/Nonlinear_Dynamic_Plant_Identification_With_Time-Varying_Characteristics/'
        experiment_name = 'Nonlinear Dynamic Plant Identification With Time-Varying Characteristics'

        dim = 2
        sigma_default = 0.3656
        delta_default = 6    
        rho_default = 0.0142
        N_default = 2
    elif dataset == MACKEY_GLASS:
        sites = ['Default']
        input_path_default = '../data/Mackey_Glass/'
        experiment_name = 'Mackey Glass'

        sigma_default = 0.127
        delta_default = 7
        rho_default = 0
        N_default = 3
    elif dataset == SP_500:
        sites = ['Default']
        input_path_default = '../data/SP_500_Daily_Closing_Price/'
        experiment_name = 'SP 500 Daily Closing Price'

        sigma_default = 0.001346
        delta_default = 33
        rho_default = 0.002251
        N_default = 1
    elif dataset == TEMPERATURE:
        sites = ["DeathValley", "Ottawa", "Lisbon"]
        input_path_default = '/home/amanda/trabalho/testes/aplicacoes/temperatura/data/'
        experiment_name = 'Wheater temperature'

        dim = 12
        sigma_default = 0.3        
        delta_default = 48
        N_default = 12
        rho_default = 1
    elif dataset == WIND:
        if mode == TEST:
            sites = ["9773", "9851", "10245", "10290", "10404", "33928", "34476", "35020", "36278", "37679", "120525", "121246", "121466", "122379", "124266"]
        else:
            sites = ["9773", "33928", "120525"]

        input_path_default = '../data/wind/'
        experiment_name = 'Wind speed'  

        dim = 2
        sigma_default = 0.2997
        delta_default = 85
        N_default = 23
        rho_default = 2.0118
    elif dataset == RAIN:
        if mode == TRAINING:
            sites = [str(i) for i in range(1, 82, 5)]
        else:
            sites = [str(i) for i in range(1, 87)]

        input_path_default = '/home/amanda/trabalho/testes/aplicacoes/precipitacao/data/'
        experiment_name = 'Rain'

        dim = 2
        sigma_default = 0.1
        delta_default = 48
        rho_default = 1
        N_default = 24
    elif dataset == GAS_FURNACE:
        sites = ['Default']
        input_path_default = '../data/gas-furnace/'
        experiment_name = 'GAS FURNACE'

        sigma_default = 0.092
        delta_default = 36
        rho_default = 2.009
        N_default = 4
        columns_ts = [0]
    elif dataset == SYSTEM_IDENTIFICATION_2:
        sites = ['Default']
        input_path_default = '../data/nonlinear_system_identification_2/'
        experiment_name = 'System Identification 2'

        sigma_default = 0.46335436826088483
        delta_default = 49
        rho_default = 0.00151647123898134
        N_default = 4
        columns_ts = [2]
    elif dataset == AUTO_MPG:
        sites = ['Default']
        input_path_default = '../data/auto-mpg/'
        experiment_name = 'Auto-MPG'

        sigma_default = 0.39714946818711083
        delta_default = 45
        rho_default = 0.14619471387405494
        N_default = 5
        columns_ts = [0,1,2,3,4,5]     
    elif dataset == AUTOS:
        sites = ['Default']
        input_path_default = '../data/autos/'
        experiment_name = 'Autos'

        sigma_default = 0.33357519994109935
        delta_default = 42
        rho_default = None
        N_default = 3
        columns_ts = None
    elif dataset == AILERONS:
        sites = ['Default']
        input_path_default = '../data/ailerons/'
        experiment_name = 'Ailerons'

        sigma_default = 0.09751084208345344
        delta_default = 76
        rho_default = 7.70706402417412
        N_default = 20
        columns_ts = None    
    elif dataset == TRIAZINES:
        sites = ['Default']
        input_path_default = '../data/triazines/'
        experiment_name = 'Triazines'

        sigma_default = 0.5592945079033025
        delta_default = 29
        rho_default = 0.001397724761260415
        N_default = 6
        columns_ts = None
    elif dataset == HELICOPTER:
        sites = ['Default']
        input_path_default = '../data/helicopter/'
        experiment_name = 'Helicopter'

        dim = 2
        sigma_default = 0.17708401024344578
        delta_default = 70
        rho_default = 0.026198431763058968
        N_default = 11
        columns_ts = None        
    elif dataset == QUADCOPTER:
        sites = ['Default']
        input_path_default = '../data/quadcopter/'
        experiment_name = 'Quadcopter'

        sigma_default = 0.07293324544725474
        delta_default = 14
        rho_default = 89.64893403045699
        N_default = 13
        columns_ts = None                

    input_path = input('Enter the dataset path (default = ' + input_path_default + '): ')
    if input_path == '':
        input_path = input_path_default

    try:
        steps_ahead = int(input('Enter the number of steps ahead: '))
    except ValueError:
        steps_ahead = 1

    experiment_name_complement = input('Add a complement for the experiment name (default = None): ')
    if experiment_name_complement != '':
        experiment_name = experiment_name + " - " + experiment_name_complement    

    sigma = list(map(float, input('Enter the sigma (default value = ' + str(sigma_default) + '): ').split()))
    if len(sigma) == 0:
        sigma = [sigma_default]

    delta = list(map(int, input('Enter the delta (default value = ' + str(delta_default) + '): ').split()))
    if len(delta) == 0:
        delta = [delta_default]

    N = list(map(int, input('Enter the size of the window (default value = ' + str(N_default) + '): ').split()))
    if len(N) == 0:
        N = [N_default]
    
    if algorithm == MTL:        
        rho = list(map(float, input('Enter the rho (default value = ' + str(rho_default) + '): ').split()))
        if len(rho) == 0:
            rho = [rho_default]
    else:
        rho = None

    register_experiment = input('Register the experiment? (default value = true): ')

    if register_experiment in ['No', 'no', 'false', 'False']:
        register_experiment = False
    else:
        register_experiment = True
    
    if dim == 2 and register_experiment:
        plot_frequency = list(map(int, input('Enter the frequency or the intervals you want to generate the plots (default = -1 in case of no plots): ').split()))
        if len(plot_frequency) == 0:
            plot_frequency = -1
        elif len(plot_frequency) != 1:
            plot = list()

            for i in range(0, len(plot_frequency), 2):
                plot = plot + list(range(plot_frequency[i], plot_frequency[i + 1]))
            
            plot_frequency = plot
    else:
        plot_frequency = -1
    
    return [algorithm, dataset, mode, sites, input_path, experiment_name, dim, sigma, delta, N, rho, register_experiment, plot_frequency, columns_ts, steps_ahead]

def run(algorithm, dataset, mode, sites, input_path, experiment_name, dim, sigma, delta, N, rho, register_experiment, plot_frequency, columns_ts, steps_ahead):
    mlflow.set_experiment(experiment_name)

    if rho is None:
        print("EVeP - " + VERSION_NAME[algorithm] + " - " + experiment_name + ": sigma = " + str(sigma) + ", delta = " + str(delta) + ", N = " + str(N))
    else:
        print("EVeP - " + VERSION_NAME[algorithm] + " - " + experiment_name + ": sigma = " + str(sigma) + ", delta = " + str(delta) + ", N = " + str(N) + ", rho = " + str(rho))

    for site in sites:        
        if dataset in [TEMPERATURE, WIND, RAIN]:
            print('Site ' + site)
            path = input_path + site + '/'
        else:
            path = input_path

        if mode == TRAINING:
            X = read_csv(path + 'X_train.csv')
            y = read_csv(path + 'Y_train.csv').reshape(-1)
        else:
            X = read_csv(path + 'X_test.csv')
            y = read_csv(path + 'Y_test.csv').reshape(-1)               

        dim = X.shape[1]

        if dim != 2:
            plot_frequency = -1

        if register_experiment:
            mlflow.start_run()    
            mlflow.set_tag("site", site)

            if plot_frequency == -1:
                mlflow.set_tag("plots", 'no')
            else:
                mlflow.set_tag("plots", 'yes')
            
            mlflow.set_tag("consequent", VERSION_NAME[algorithm])

            if mode == TRAINING:
                mlflow.set_tag("mode", "training")
            else:
                mlflow.set_tag("mode", "test")

            artifact_uri = mlflow.get_artifact_uri()
            # removing the 'file://'
            artifact_uri = artifact_uri[7:] + '/'            

            mlflow.log_param("sigma", sigma)
            mlflow.log_param("delta", delta)
            mlflow.log_param("N", N)

            if algorithm == MTL:
                mlflow.log_param("rho", rho)

        model = EVeP(sigma, delta, N, rho, columns_ts)

        predictions = np.zeros((y.shape[0], 1))
        number_of_rules = np.zeros((y.shape[0], 1))
        RMSE = np.zeros((y.shape[0], 1))
        time_ = np.zeros((y.shape[0], 1))
        queue_X = []
        queue_Y = []

        for i in tqdm(range(y.shape[0])):
            start = time.time()
            predictions[i, 0] = model.predict(X[i, :].reshape(1, -1))
            queue_X.append(X[i, :].reshape(1, -1)), queue_Y.append(y[i].reshape(1, -1))

            if i >= steps_ahead - 1:
                model.train(queue_X.pop(0).reshape(1, -1), queue_Y.pop(0).reshape(1, -1), np.array([[i]]))

            end = time.time()
            time_[i, 0] = end - start

            # Saving statistics for the step i
            number_of_rules[i, 0] = model.c
            
            RMSE[i, 0] = mean_squared_error(y[:i+1], predictions[:i+1], squared=False)

            if plot_frequency != -1: 
                if len(plot_frequency) == 1:
                    if (i % plot_frequency[0]) == 0:
                        model.plot(artifact_uri + str(i) + '_input.png', artifact_uri + str(i) + '_output.png', i)
                elif i in plot_frequency:
                    model.plot(artifact_uri + str(i) + '_input.png', artifact_uri + str(i) + '_output.png', i)

        if register_experiment:
            error = (y - predictions[:, 0]).reshape(-1, 1)
            np.savetxt(artifact_uri + 'predictions.csv', predictions)
            np.savetxt(artifact_uri + 'rules.csv', number_of_rules)
            np.savetxt(artifact_uri + 'error.csv', error)
            np.savetxt(artifact_uri + 'time.csv', time_)

            plot_graph(number_of_rules, 'Number of rules', 'Step', artifact_uri + 'rules.png')
            plot_graph(RMSE, 'RMSE', 'Step', artifact_uri + 'RMSE.png')
            plot_graph(error, 'Error', 'Step', artifact_uri + 'Error.png')
            plot_graph(y, 'Prediction', 'Step', artifact_uri + 'predictions.png', predictions, "y", "p")        

            mlflow.log_metric('RMSE', RMSE[-1, 0])
            mlflow.log_metric('NDEI', RMSE[-1, 0] / np.std(y))            
            mlflow.log_metric('Mean_rules', np.mean(number_of_rules))
            mlflow.log_metric('Last_No_rule', number_of_rules[-1, 0])

            if algorithm != MTL:
                # x0, y0, shape_x, scale_x, shape_y, scale_y, theta, hr, sigma, tau, window
                mlflow.log_metric('Last_No_parameters', number_of_rules[-1, 0] * (6 + dim + 1) + 4)
                mlflow.log_metric('Mean_parameters', np.mean(number_of_rules) * (6 + dim + 1) + 4)
            else:
                # x0, y0, shape_x, scale_x, shape_y, scale_y, theta, hr, sigma, tau, window, rho, rho_2, rho_3
                mlflow.log_metric('Last_No_parameters', number_of_rules[-1, 0] * (6 + dim + 1) + 7)
                mlflow.log_metric('Mean_parameters', np.mean(number_of_rules) * (6 + dim + 1) + 7)    

            mlflow.end_run()        

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    abspath = os.path.abspath(__file__)
    os.chdir(os.path.dirname(abspath)) 

    [algorithm, dataset, mode, site, input_path, experiment_name, dim, sigmas, refresh_rates, window_sizes, rho_1s, register_experiment, plot_frequency, columns_ts, steps_ahead] = read_parameters()

    for sigma in sigmas:
        for delta in refresh_rates:
            for N in window_sizes:
                if algorithm == MTL:
                    for rho in rho_1s:
                        run(algorithm, dataset, mode, site, input_path, experiment_name, dim, sigma, delta, N, rho, register_experiment, plot_frequency, columns_ts, steps_ahead)
                else:
                    run(algorithm, dataset, mode, site, input_path, experiment_name, dim, sigma, delta, N, rho_1s, register_experiment, plot_frequency, columns_ts, steps_ahead)