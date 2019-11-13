import csv
from eEVM import eEVM
from eEVM_RLS import eEVM_RLS
from eEVM_RLS_mod import eEVM_RLS_mod
from eEVM_MTL import eEVM_MTL
from math import sqrt
import matplotlib.pyplot as plt
import mlflow
import numpy as np
from sklearn.metrics import mean_squared_error
from tqdm import tqdm

# databases IDs
PLANT_IDENTIFICATION = 1
MACKEY_GLASS = 2
SP_500 = 3
TEMPERATURE = 4
WIND = 5

# algorithm versions
BATCH = 0
RLS = 1
RLS_MOD = 2
MTL = 3

VERSION_NAME = ['BATCH', 'RLS', 'RLS_MOD', 'MTL']

TRAINING = 0
TEST = 1

def read_csv(file, dataset):
    database = []

    if dataset in [TEMPERATURE, PLANT_IDENTIFICATION, MACKEY_GLASS, SP_500]:
        delimiter = ','
    elif dataset == WIND:
        delimiter = ' '

    with open(file, newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=delimiter, quotechar='|')
        for row in spamreader:
            database.append(row)

    return np.asarray(database).astype('float')

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

def plot_graph_2_scales(y, y_label, x_label, file_name, y_aux, y_aux_label):
    fig, ax1 = plt.subplots()

    color = 'tab:orange'
    ax1.set_xlabel(x_label)
    ax1.set_ylabel(y_label, color=color)
    ax1.plot(y, color=color)
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

    color = 'tab:blue'
    ax2.set_ylabel(y_aux_label, color=color)  # we already handled the x-label with ax1
    ax2.plot(y_aux, color=color)
    ax2.tick_params(axis='y', labelcolor=color)

    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.savefig(file_name)
    plt.close()

def read_parameters():
    try:
        algorithm = int(input('Enter the version of eEVM to run:\n1- Batch\n2- RLS\n3- RLS_mod\n4- MTL (default)\n')) - 1
    except ValueError:
        algorithm = MTL

    try:
        mode = int(input('Run the 1- training or 2- test (default) dataset?\n')) - 1
    except ValueError:
        mode = TEST  

    try:
        dataset = int(input('Enter the dataset to be tested:\n1- Nonlinear Dynamic Plant Identification With Time-Varying Characteristics (default)\n' + 
        '2- Mackey–Glass Chaotic Time Series (Long-Term Prediction)\n3- Online Prediction of S&P 500 Daily Closing Price\n' + 
        '4- Wheater temperature\n5- Wind speed\n'))
    except ValueError:
        dataset = PLANT_IDENTIFICATION

    if dataset == PLANT_IDENTIFICATION:
        sites = ['Default']
        input_path = '/home/amanda/Dropbox/trabalho/doutorado/testes/aplicacoes/Nonlinear_Dynamic_Plant_Identification_With_Time-Varying_Characteristics/'
        experiment_name = 'Nonlinear Dynamic Plant Identification With Time-Varying Characteristics'
    elif dataset == MACKEY_GLASS:
        sites = ['Default']
        input_path = '/home/amanda/Dropbox/trabalho/doutorado/testes/aplicacoes/Mackey_Glass/'
        experiment_name = 'Mackey Glass'
    elif dataset == SP_500:
        sites = ['Default']
        input_path = '/home/amanda/Dropbox/trabalho/doutorado/testes/aplicacoes/SP_500_Daily_Closing_Price/'
        experiment_name = 'SP 500 Daily Closing Price'
    elif dataset == TEMPERATURE:
        sites = ["DeathValley", "Ottawa", "Lisbon"]
        input_path = '/home/amanda/Dropbox/trabalho/doutorado/testes/aplicacoes/temperatura/'
        experiment_name = 'Wheater temperature'
    else:
        if mode == TEST:
            sites = ["9773", "9851", "10245", "10290", "10404", "33928", "34476", "35020", "36278", "37679", "120525", "121246", "121466", "122379", "124266"]
        else:
            sites = ["9773", "33928", "120525"]
        input_path = '/home/amanda/Dropbox/trabalho/doutorado/testes/aplicacoes/vento/USA/'
        experiment_name = 'Wind speed'  

    experiment_name_complement = input('Add a complement for the experiment name (default = None): ')
    if experiment_name_complement != '':
        experiment_name = experiment_name + " - " + experiment_name_complement    

    if dataset == TEMPERATURE or dataset == WIND:
        try:
            dim = int(input('Enter the number of dimensions of the input (default value = 12): '))
        except ValueError:
            dim = 12
    else:
        dim = -1

    sigma = list(map(float, input('Enter the sigma (default value = 0.5): ').split()))
    if len(sigma) == 0:
        sigma = [0.5]

    tau = list(map(int, input('Enter the tau (default value = 99999): ').split()))
    if len(tau) == 0:
        tau = [99999]

    refresh_rate = list(map(int, input('Enter the refresh_rate (default value = 50): ').split()))
    if len(refresh_rate) == 0:
        refresh_rate = [50]

    window_size = list(map(int, input('Enter the size of the window (default value = 4): ').split()))
    if len(window_size) == 0:
        window_size = [4]
    
    if algorithm == MTL:        
        rho_1 = list(map(float, input('Enter the rho_1 (default value = 1): ').split()))
        if len(rho_1) == 0:
            rho_1 = [1]

        rho_2 = list(map(float, input('Enter the rho_2 (default value = 0): ').split()))
        if len(rho_2) == 0:
            rho_2 = [0]

        rho_3 = list(map(float, input('Enter the rho_3 (default value = 0): ').split()))
        if len(rho_3) == 0:
            rho_3 = [0]

        thr_sigma = list(map(float, input('Enter the thr_sigma (default value = 0.1): ').split()))
        if len(thr_sigma) == 0:
            thr_sigma = [0.1]
    else:
        rho_1 = None
        rho_2 = None
        rho_3 = None
        thr_sigma = None

    register_experiment = input('Register the experiment? (default value = true): ')

    if register_experiment in ['No', 'no', 'false', 'False']:
        register_experiment = False
    else:
        register_experiment = True
    
    if register_experiment:
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
    
    return [algorithm, dataset, mode, sites, input_path, experiment_name, dim, sigma, tau, refresh_rate, window_size, rho_1, rho_2, rho_3, thr_sigma, register_experiment, plot_frequency]

def run(algorithm, dataset, mode, sites, input_path, experiment_name, dim, sigma, tau, refresh_rate, window_size, rho_1, rho_2, rho_3, thr_sigma, register_experiment, plot_frequency):
    mlflow.set_experiment(experiment_name)

    if rho_1 is None:
        print("eEVM_" + VERSION_NAME[algorithm] + " - " + experiment_name + ": sigma = " + str(sigma) + ", tau = " + str(tau) + ", refresh_rate = " + str(refresh_rate) + ", window_size = " + str(window_size))
    else:
        print("eEVM_" + VERSION_NAME[algorithm] + " - " + experiment_name + ": sigma = " + str(sigma) + ", tau = " + str(tau) + ", refresh_rate = " + str(refresh_rate) + ", window_size = " + str(window_size) + ", rho_1 = " + str(rho_1) + ", rho_2 = " + str(rho_2) + ", rho_3 = " + str(rho_3) + ", thr_sigma = " + str(thr_sigma))

    for site in sites:
        print('Site ' + site)    
        
        if dataset in [PLANT_IDENTIFICATION, MACKEY_GLASS, SP_500]:
            if mode == TRAINING:
                X = read_csv(input_path + 'base/X_train.csv', dataset)
                y = read_csv(input_path + 'base/Y_train.csv', dataset)
            else:
                X = read_csv(input_path + 'base/X_test.csv', dataset)
                y = read_csv(input_path + 'base/Y_test.csv', dataset)                

        elif dataset in [TEMPERATURE, WIND]:
            if dataset == TEMPERATURE:
                path = input_path + 'bases/' + site + '/' + str(dim)
            else:
                if mode == TRAINING:
                    path = input_path + 'bases/' + site + '/wind_speed/hour/' + site + '-2011/' + str(dim)
                else:
                    path = input_path + 'bases/' + site + '/wind_speed/hour/' + site + '-2012/' + str(dim)

            X  = read_csv(path + '/X_real.csv', dataset)
            y  = read_csv(path + '/Y_real.csv', dataset).reshape(-1)        

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
            
            mlflow.set_tag("alg", "eEVM_" + VERSION_NAME[algorithm])

            if mode == TRAINING:
                mlflow.set_tag("mode", "training")
            else:
                mlflow.set_tag("mode", "test")

            artifact_uri = mlflow.get_artifact_uri()
            # removing the 'file://'
            artifact_uri = artifact_uri[7:] + '/'            

            mlflow.log_param("dim", dim)
            mlflow.log_param("sigma", sigma)
            mlflow.log_param("tau", tau)
            mlflow.log_param("refresh_rate", refresh_rate)
            mlflow.log_param("window_size", window_size)

            if algorithm == MTL:
                mlflow.log_param("rho_1", rho_1)
                mlflow.log_param("rho_2", rho_2)
                mlflow.log_param("rho_3", rho_3)
                mlflow.log_param("thr_sigma", thr_sigma)

        if algorithm == BATCH:
            model = eEVM(sigma, tau, refresh_rate, window_size)
        elif algorithm == RLS:
            model = eEVM_RLS(sigma, tau, refresh_rate, window_size)
        elif algorithm == RLS_MOD:
            model = eEVM_RLS_mod(sigma, tau, refresh_rate, window_size)        
        elif algorithm == MTL:
            model = eEVM_MTL(sigma, tau, refresh_rate, window_size, rho_1, rho_2, rho_3, thr_sigma)

        predictions = np.zeros((y.shape[0], 1))
        number_of_clusters = np.zeros((y.shape[0], 1))
        number_of_EVs = np.zeros((y.shape[0], 1))        
        RMSE = np.zeros((y.shape[0], 1))

        if algorithm == MTL:
            connection_rate = np.zeros((y.shape[0], 1))

        for i in tqdm(range(y.shape[0])):
            predictions[i, 0] = model.predict(X[i, :].reshape(1, -1))
            model.train(X[i, :].reshape(1, -1), y[i].reshape(1, -1), np.array([[i]]))

            # Saving statistics for the step i
            number_of_clusters[i, 0] = model.get_number_of_clusters()
            number_of_EVs[i, 0] = model.c

            if algorithm == MTL:
                connection_rate[i, 0] = model.connection_rate
            
            if i == 0:
                RMSE[i, 0] = sqrt(mean_squared_error(y[:i+1], predictions[:i+1]))
            else:
                # Desconsider the first prediction because there was no previous model 
                RMSE[i, 0] = sqrt(mean_squared_error(y[1:i+1], predictions[1:i+1]))

            if plot_frequency != -1: 
                if len(plot_frequency) == 1:
                    if (i % plot_frequency[0]) == 0:
                        model.plot(artifact_uri + str(i) + '.png')
                elif i in plot_frequency:
                    model.plot(artifact_uri + str(i) + '.png')

        if register_experiment:
            np.savetxt(artifact_uri + 'predictions.csv', predictions)
            np.savetxt(artifact_uri + 'clusters.csv', number_of_clusters)
            np.savetxt(artifact_uri + 'EVs.csv', number_of_EVs)            

            plot_graph(number_of_clusters, 'Quantity', 'Step', artifact_uri + 'rules.png', number_of_EVs, "Number of clusters", 'Number of EVs')
            plot_graph(RMSE, 'RMSE', 'Step', artifact_uri + 'RMSE.png')
            plot_graph(y, 'Prediction', 'Step', artifact_uri + 'predictions.png', predictions, "y", "p")        

            mlflow.log_metric('RMSE', RMSE[-1, 0])
            mlflow.log_metric('NDEI', RMSE[-1, 0] / np.std(y))            
            mlflow.log_metric('Mean_clusters', np.mean(number_of_clusters))        
            mlflow.log_metric('Mean_EVs', np.mean(number_of_EVs))
            mlflow.log_metric('Last_No_EV', number_of_EVs[-1, 0])

            if algorithm != MTL:
                # x0, y0, shape_x, scale_x, shape_y, scale_y, theta, hr, sigma, tau, window
                mlflow.log_metric('Last_No_parameters', number_of_EVs[-1, 0] * (6 + dim + 1) + 4)
                mlflow.log_metric('Mean_parameters', np.mean(number_of_EVs) * (6 + dim + 1) + 4)
            else:
                # x0, y0, shape_x, scale_x, shape_y, scale_y, theta, hr, sigma, tau, window, rho_1, rho_2, rho_3
                mlflow.log_metric('Last_No_parameters', number_of_EVs[-1, 0] * (6 + dim + 1) + 7)
                mlflow.log_metric('Mean_parameters', np.mean(number_of_EVs) * (6 + dim + 1) + 7)    

                np.savetxt(artifact_uri + 'connection_rate.csv', connection_rate)
                plot_graph_2_scales(number_of_EVs, 'Number of EVs', 'Step', artifact_uri + 'connection_rate.png', connection_rate, 'Connection Rate')

            mlflow.end_run()        

if __name__ == "__main__":
    [algorithm, dataset, mode, site, input_path, experiment_name, dim, sigmas, taus, refresh_rates, window_sizes, rho_1s, rho_2s, rho_3s, thr_sigmas, register_experiment, plot_frequency] = read_parameters()

    for sigma in sigmas:
        for tau in taus:
            for refresh_rate in refresh_rates:
                for window_size in window_sizes:
                    if algorithm == MTL:
                        for rho_1 in rho_1s:
                            for rho_2 in rho_2s:
                                for rho_3 in rho_3s:
                                    for thr_sigma in thr_sigmas:
                                        run(algorithm, dataset, mode, site, input_path, experiment_name, dim, sigma, tau, refresh_rate, window_size, rho_1, rho_2, rho_3, thr_sigma, register_experiment, plot_frequency)
                    else:
                        run(algorithm, dataset, mode, site, input_path, experiment_name, dim, sigma, tau, refresh_rate, window_size, rho_1s, rho_2s, rho_3s, thr_sigmas, register_experiment, plot_frequency)