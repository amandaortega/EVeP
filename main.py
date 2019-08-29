import csv
from eEVM import eEVM
from math import sqrt
import matplotlib.pyplot as plt
import mlflow
import numpy as np
from sklearn.metrics import mean_squared_error
from tqdm import tqdm

def read_csv(file, dataset):
    database = []

    if dataset == 1:
        delimiter = ','
    else:
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

try:
    dataset = int(input('Enter the dataset to be tested:\n1- Wheater temperature\n2- Wind speed (default)\n'))
except ValueError:
    dataset = 2

if dataset == 1:
    sites = ["DeathValley", "Ottawa", "Lisbon"]
    input_path = '/home/amanda/Dropbox/trabalho/doutorado/testes/aplicacoes/temperatura/'
    mlflow.set_experiment('Wheater temperature')
else:
    sites = ["9773", "9851", "10245", "10290", "10404", "33928", "34476", "35020", "36278", "37679", "120525", "121246", "121466", "122379", "124266"]
    input_path = '/home/amanda/Dropbox/trabalho/doutorado/testes/aplicacoes/vento/USA/'
    mlflow.set_experiment('Wind speed')

try:
    dim = int(input('Enter the number of dimensions of the input (default value = 2): '))
except ValueError:
    dim = 2

if dim == 2:
    try:
        plot_frequency = int(input('Enter the frequency to generate the plots (-1 in case of no plots, default = 50): '))
    except ValueError:
        plot_frequency = -1
else:
    plot_frequency = -1

try:
    sigma = int(input('Enter the sigma (default value = 0.5): '))
except ValueError:
    sigma = 0.5

try:
    tau = int(input('Enter the tau (default value = 75): '))
except ValueError:
    tau = 75

try:
    refresh_rate = int(input('Enter the refresh_rate (default value = 50): '))
except ValueError:
    refresh_rate = 50

try:
    window_size = int(input('Enter the size of the window (default value = 50): '))
except ValueError:
    window_size = 50

register_experiment = input('Register the experiment? (default value = true): ')

if register_experiment in ['No', 'no', 'false', 'False']:
    register_experiment = False
else:
    register_experiment = True

for site in sites:
    print('Site ' + site)    
    
    if dataset == 1:
        path = input_path + 'bases/' + site + '/' + str(dim)
    else:
        path = input_path + 'bases/' + site + '/wind_speed/hour/' + site + '-2012/' + str(dim)

    X  = read_csv(path + '/X_real.csv', dataset)
    y  = read_csv(path + '/Y_real.csv', dataset).reshape(-1)        

    X_min = read_csv(path + '/XSuppInf.csv', dataset)
    X_max = read_csv(path + '/XSuppSup.csv', dataset)
    
    y_min = read_csv(path + '/YSuppInf.csv', dataset)
    y_max = read_csv(path + '/YSuppSup.csv', dataset)    

    if register_experiment:
        mlflow.start_run()    
        mlflow.set_tag("site", site)

        if plot_frequency == -1:
            mlflow.set_tag("plots", 'no')
        else:
            mlflow.set_tag("plots", 'yes')

        artifact_uri = mlflow.get_artifact_uri()
        # removing the 'file://'
        artifact_uri = artifact_uri[7:] + '/'

        mlflow.log_param("dim", dim)
        mlflow.log_param("sigma", sigma)
        mlflow.log_param("tau", tau)
        mlflow.log_param("refresh_rate", refresh_rate)
        mlflow.log_param("window_size", window_size)

    model = eEVM(sigma, tau, refresh_rate, window_size)

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
        
        mlflow.log_metric('RMSE', RMSE[-1, 0])
        mlflow.log_metric('Mean_clusters', np.mean(number_of_clusters))
        mlflow.log_metric('Mean_EVs', np.mean(number_of_EVs))

        mlflow.end_run()