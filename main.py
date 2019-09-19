import csv
from eEVM import eEVM
from eEVM_RLS import eEVM_RLS
from eEVM_RLS_mod import eEVM_RLS_mod
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
BATCH = 1
RLS = 2
RLS_MOD = 3

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

def read_parameters():
    try:
        algorithm = int(input('Enter the version of eEVM to run:\n1- Batch\n2- RLS\n3- RLS_mod (default)\n'))
    except ValueError:
        algorithm = RLS_MOD

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
        sites = ["9773", "9851", "10245", "10290", "10404", "33928", "34476", "35020", "36278", "37679", "120525", "121246", "121466", "122379", "124266"]
        input_path = '/home/amanda/Dropbox/trabalho/doutorado/testes/aplicacoes/vento/USA/'
        experiment_name = 'Wind speed'

    if dataset == TEMPERATURE or dataset == WIND:
        try:
            dim = int(input('Enter the number of dimensions of the input (default value = 12): '))
        except ValueError:
            dim = 12
    else:
        dim = -1

    try:
        sigma = float(input('Enter the sigma (default value = 0.5): '))
    except ValueError:
        sigma = 0.5

    try:
        tau = int(input('Enter the tau (default value = 300): '))
    except ValueError:
        tau = 300

    try:
        refresh_rate = int(input('Enter the refresh_rate (default value = 50): '))
    except ValueError:
        refresh_rate = 50

    try:
        window_size = int(input('Enter the size of the window (default value = 4): '))
    except ValueError:
        window_size = 4

    register_experiment = input('Register the experiment? (default value = true): ')

    if register_experiment in ['No', 'no', 'false', 'False']:
        register_experiment = False
    else:
        register_experiment = True
    
    try:
        plot_frequency = int(input('Enter the frequency to generate the plots (default = -1 in case of no plots): '))
    except ValueError:
        plot_frequency = -1
    
    return [algorithm, dataset, sites, input_path, experiment_name, dim, sigma, tau, refresh_rate, window_size, register_experiment, plot_frequency]

def run(algorithm, dataset, sites, input_path, experiment_name, dim, sigma, tau, refresh_rate, window_size, register_experiment, training, plot_frequency):
    mlflow.set_experiment(experiment_name)

    for site in sites:
        print('Site ' + site)    
        
        if dataset in [PLANT_IDENTIFICATION, MACKEY_GLASS, SP_500]:
            if training:
                X = read_csv(input_path + 'base/X_train.csv', dataset)
                y = read_csv(input_path + 'base/Y_train.csv', dataset)
            else:
                X = read_csv(input_path + 'base/X_test.csv', dataset)
                y = read_csv(input_path + 'base/Y_test.csv', dataset)                

            X_min = X
            X_max = X
                            
            y_min = y
            y_max = y                

        elif dataset in [TEMPERATURE, WIND]:
            if dataset == TEMPERATURE:
                path = input_path + 'bases/' + site + '/' + str(dim)
            else:
                if training:
                    path = input_path + 'bases/' + site + '/wind_speed/hour/' + site + '-2011/' + str(dim)
                else:
                    path = input_path + 'bases/' + site + '/wind_speed/hour/' + site + '-2012/' + str(dim)

            X  = read_csv(path + '/X_real.csv', dataset)
            y  = read_csv(path + '/Y_real.csv', dataset).reshape(-1)        

            X_min = read_csv(path + '/XSuppInf.csv', dataset)
            X_max = read_csv(path + '/XSuppSup.csv', dataset)
            
            y_min = read_csv(path + '/YSuppInf.csv', dataset)
            y_max = read_csv(path + '/YSuppSup.csv', dataset)    

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
            
            if algorithm == BATCH:
                mlflow.set_tag("alg", "eEVM_batch")
            elif algorithm == RLS:
                mlflow.set_tag("alg", "eEVM_RLS")
            elif algorithm == RLS_MOD:
                mlflow.set_tag("alg", "eEVM_RLS_mod")                

            artifact_uri = mlflow.get_artifact_uri()
            # removing the 'file://'
            artifact_uri = artifact_uri[7:] + '/'

            mlflow.log_param("dim", dim)
            mlflow.log_param("sigma", sigma)
            mlflow.log_param("tau", tau)
            mlflow.log_param("refresh_rate", refresh_rate)
            mlflow.log_param("window_size", window_size)

        if algorithm == BATCH:
            model = eEVM(sigma, tau, refresh_rate, window_size)
        elif algorithm == RLS:
            model = eEVM_RLS(sigma, tau, refresh_rate, window_size)
        elif algorithm == RLS_MOD:
            model = eEVM_RLS_mod(sigma, tau, refresh_rate, window_size)        

        predictions = np.zeros((y.shape[0], 1))
        number_of_clusters = np.zeros((y.shape[0], 1))
        number_of_EVs = np.zeros((y.shape[0], 1))
        RMSE = np.zeros((y.shape[0], 1))

        for i in tqdm(range(y.shape[0])):
            predictions[i, 0] = model.predict(X[i, :].reshape(1, -1))        
            model.train(X_min[i, :].reshape(1, -1), X[i, :].reshape(1, -1), X_max[i, :].reshape(1, -1), y_min[i].reshape(1, -1), y[i].reshape(1, -1), y_max[i].reshape(1, -1), np.array([[i]]))

            # Saving statistics for the step i
            number_of_clusters[i, 0] = model.get_number_of_clusters()
            number_of_EVs[i, 0] = model.get_number_of_EVs()
            
            if i == 0:
                RMSE[i, 0] = sqrt(mean_squared_error(y[:i+1], predictions[:i+1]))
            else:
                # Desconsider the first prediction because there was no previous model 
                RMSE[i, 0] = sqrt(mean_squared_error(y[1:i+1], predictions[1:i+1]))

            if plot_frequency != -1: 
                if (i % plot_frequency) == 0:
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

            mlflow.end_run()        

if __name__ == "__main__":
    [algorithm, dataset, site, input_path, experiment_name, dim, sigma, tau, refresh_rate, window_size, register_experiment, plot_frequency] = read_parameters()
    run(algorithm, dataset, site, input_path, experiment_name, dim, sigma, tau, refresh_rate, window_size, register_experiment, False, plot_frequency)