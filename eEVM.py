"""
    Author: Amanda Ortega de Castro Ayres
    Created in: June 25, 2019
    Python version: 3.6
"""

import libmr
from matplotlib import pyplot, cm
from matplotlib.patches import Circle
from mpl_toolkits.mplot3d import Axes3D, art3d
import numpy as np
from random import random
from sklearn.linear_model import LinearRegression
import sklearn.metrics
from time import time
from tqdm import tqdm

class eEVM(object):

    """
    evolving Extreme Value Machine
    Ruled-based predictor with EVM at the definition of the antecedent of the rules.    
    1. Create a new instance and provide the model parameters;
    2. Call the predict(x) method to make predictions based on the given input;
    3. Call the evolve(x, y) method to evolve the model based on the new input-output pair.
    """

    class Cluster(object):
        """
        Cluster grouping near samples.
        """
        
        class EV(object):
            """
            Extreme vector.
            """

            # Initialization of a new instance of EV.
            def __init__(self, x0, y0, tau):
                self.mr_x = libmr.MR()
                self.mr_y = libmr.MR()
                self.mr_xy = libmr.MR()
                self.tau = tau
                self.x0 = x0
                self.y0 = y0
                self.X = x0
                self.y = y0

            # Add the sample(s) (X, y) as covered by the extreme vector
            def add_sample(self, X, y):
                self.X = np.concatenate((self.X, X))                
                self.y = np.concatenate((self.y, y))

            # Calculate the firing degree of the sample to the psi curve
            def firing_degree(self, x=None, y=None):
                if y is None:
                    return self.mr_x.w_score(sklearn.metrics.pairwise.pairwise_distances(self.x0, x))
                elif x is None:
                    return self.mr_y.w_score(sklearn.metrics.pairwise.pairwise_distances(self.y0, y))
                else:
                    return self.mr_xy.w_score(sklearn.metrics.pairwise.pairwise_distances(np.concatenate((self.x0, self.y0), axis=1), np.concatenate((x, y), axis=1)))

            # Fit the psi curve to the extreme values with distance D to the center of the EV
            def fit_x(self, D):
                self.mr_x.fit_low(1/2 * D, min(D.shape[0], self.tau))   

            # Fit the psi curve to the extreme values with distance D to the center of the EV
            def fit_y(self, D):
                self.mr_y.fit_low(1/2 * D, min(D.shape[0], self.tau))                   

            # Fit the psi curve to the extreme values with distance D to the center of the EV
            def fit_xy(self, D):
                self.mr_xy.fit_low(1/2 * D, min(D.shape[0], self.tau))                                   

            # Get the distance from the origin of the EV which has the given probability to belong to the curve
            def get_distance(self, percentage):
                return self.mr_x.inv(percentage)

            # Plot the probability of sample inclusion (psi-model) together with the samples associated with the EV
            def plot(self, ax, marker, color, z_bottom, sigma):
                # Plot the input samples in the XY plan
                sc = ax.scatter(self.X[:, 0], self.X[:, 1], z_bottom * np.ones((self.X.shape[0], 1)), marker=marker, color=color)

                # Plot the radius for which there is a probability sigma to belong to the EV
                radius = self.get_distance(sigma)
                p = Circle((self.x0[0, 0], self.x0[0, 1]), radius, fill=False, color=color)
                ax.add_patch(p)
                art3d.pathpatch_2d_to_3d(p, z=z_bottom, zdir="z")

                # Plot the psi curve of the EV
                r = np.linspace(0, self.get_distance(0.05), 100)
                theta = np.linspace(0, 2 * np.pi, 145)    
                radius_matrix, theta_matrix = np.meshgrid(r,theta)            
                X = self.x0[0, 0] + radius_matrix * np.cos(theta_matrix)
                Y = self.x0[0, 1] + radius_matrix * np.sin(theta_matrix)
                points = np.array([np.array([X, Y])[0, :, :].reshape(-1), np.array([X, Y])[1, :, :].reshape(-1)]).T
                Z = self.mr_x.w_score_vector(sklearn.metrics.pairwise.pairwise_distances(self.x0, points).reshape(-1))                
                ax.plot_surface(X, Y, Z.reshape((X.shape[0], X.shape[1])), antialiased=False, cmap=cm.coolwarm, alpha=0.1)

            # Predict the local output of x based on the linear regression of the samples stored at the EV
            def predict(self, x):
                return LinearRegression().fit(self.X, self.y).predict(x)

        def __init__(self, tau=75, sigma=0.5):
            self.EVs = list()
            self.tau = tau
            self.sigma = sigma

        def add_EV(self, x0, y0, X_ext, y_ext):
            self.EVs.append(self.EV(x0, y0, self.tau))
            self.EVs[-1].fit_x(sklearn.metrics.pairwise.pairwise_distances(x0, X_ext)[0])
            self.EVs[-1].fit_y(sklearn.metrics.pairwise.pairwise_distances(y0, y_ext)[0])
            self.EVs[-1].fit_xy(sklearn.metrics.pairwise.pairwise_distances(np.concatenate((x0, y0), axis=1), np.concatenate((X_ext, y_ext), axis=1))[0])

        def firing_degrees(self, x, y):
            return np.array([ev.firing_degree(x, y) for ev in self.EVs])
        
        def get_X(self):
            return np.concatenate([ev.X for ev in self.EVs])

        def get_y(self):        
            return np.concatenate([ev.y for ev in self.EVs])

        # Plot the probability of sample inclusion (psi-model) for each extreme value
        def plot(self, ax, marker, color, z_bottom):
            for ev in self.EVs:
                ev.plot(ax, marker, color, z_bottom, self.sigma)

        def refresh(self, X_ext, y_ext):
            X_in = self.get_X()
            y_in = self.get_y()

            self.EVs = list()
            EVs_temp = list()

            # calcula os pares de distâncias entre as amostras da classe atual, X_l, e as amostras das outras classes, X_m
            D = sklearn.metrics.pairwise.pairwise_distances(X_in, X_ext)
            D_y = sklearn.metrics.pairwise.pairwise_distances(np.concatenate((X_in, y_in), axis=1), np.concatenate((X_ext, y_ext), axis=1))

            # para cada amostra pertencente à classe Cl, estima os parâmetros shape e scale com base na metade da distância
            # das tau amostras mais próximas não pertencentes a Cl
            for i in range(X_in.shape[0]):
                EVs_temp.append(self.EV(X_in[i, :], y_in[i], self.tau))
                EVs_temp[-1].fit_x(D[i])
                EVs_temp[-1].fit_y(D_y[i])   

            Nl = X_in.shape[0]
            U = range(Nl)
            S = np.zeros((Nl, Nl))

            # percorre todas as amostras e verifica se a probabilidade gerada pela função psi entre a distância entre cada par 
            # de pontos é maior ou igual a sigma
            for i in U:
                for j in U:
                    if self.EVs[i].firing_degree(X_in[j, :]) >= self.sigma:
                        S[i, j] = 1
            
            C = []
            I = []

            # enquanto os pontos representados pelos valores extremos não abrangerem o universo das amostras
            while (set(C) != set(U)):
                # manipulações necessárias para que a matriz diferenca contenha 1 apenas quando há um 
                # ponto ainda não coberto e que é representado por um valor extremo candidato
                aux = np.zeros(Nl)
                aux[C] = 1
                diferenca = S - np.matlib.repmat(aux, Nl, 1)
                diferenca[I] = np.zeros((len(I), Nl))
                diferenca[:, I] = np.zeros((Nl, len(I)))
                diferenca = np.clip(diferenca, 0, 1)
                
                # obtém o valor extremo que representa a maior quantidade de pontos ainda não cobertos
                ind = np.argmax(np.sum(diferenca, axis=1), axis=0)

                # acrescenta os novos pontos cobertos
                new_points = np.asarray(np.where(S[ind])).reshape(-1)
                C = np.append(C, new_points)
                C = C.astype(int)
                C = np.unique(C)

                # add the samples covered by the EV
                EVs_temp[ind].add_sample(X_in[new_points], y_in[new_points])

                # acrescenta o novo valor extremo ao conjunto
                self.EVs.append(EVs_temp[ind])

    # Model initialization
    def __init__(self, sigma=0.5, tau=75):
        # Setting rule base history
        self.number_of_rules = list()        

        # Setting local models
        self.models = list()

        # Setting EVM algorithm parameters
        self.sigma = sigma
        self.tau = tau

    # Plot the granules that form the antecedent part of the rules
    def plot(self):
        fig = pyplot.figure()
        ax = fig.add_subplot(111, projection='3d')
        z_bottom = -0.3
        ax.set_zlim(bottom=z_bottom, top=1)
        ax.set_xlim(left=0, right=1)
        ax.set_ylim(bottom=0, top=1)
        ax.set_zticklabels("")        

        colors = cm.get_cmap('hsv', len(self.models))

        for i, m in enumerate(self.models):
            m.plot(ax, '.', colors(i), z_bottom)

        pyplot.show()

    # Predict the output given the input sample x
    def predict(self, x):
        # Checking for system prior knowledge
        if len(self.models) == 0:
            return random()

        num = 0
        den = 0

        for cluster in self.models:
            for EV in cluster.EVs:
                num = num + EV.firing_degree(x) * EV.predict(x)
                den = den + EV.firing_degree(x)

        if den == 0:
            return random()

        # Calculating the output
        output = num / den

        if output[0] > 1:
            return 1./np.absolute(output[0])
        elif output[0] < 0:
            return np.absolute(output[0])
        return output[0]          

    # Evolves the model (main method)
    def train(self, x_min, x, x_max, y_min, y, y_max):
        # empty antecedents
        if len(self.models) == 0:
            self.models.append(self.Cluster(self.tau, self.sigma))
            self.models[-1].add_EV(x, y, np.concatenate((x_min, x_max), axis=0), np.concatenate((y_min, y_max), axis=0))
        else:
            best_EV = None
            best_EV_value = 0
            
            best_model_y = None
            best_EV_y_value = 0            

            # check if it is possible to insert the sample in an existing model
            for m in self.models:
                firing_degrees = m.firing_degrees(x, y)
                best_index = np.argmax(firing_degrees)

                firing_degrees_y = m.firing_degrees(None, y)
                best_index_y = np.argmax(firing_degrees_y)                

                if firing_degrees[best_index] > best_EV_value and firing_degrees[best_index] > self.sigma:
                    best_EV = m.EVs[best_index]
                    best_EV_value = firing_degrees[best_index]
                elif firing_degrees_y[best_index_y] > best_EV_y_value and firing_degrees_y[best_index_y] > self.sigma:
                    best_model_y = m
                    best_EV_y_value = firing_degrees_y[best_index_y]                    
            
            if best_EV is not None:
                best_EV.add_sample(x, y)
            else:
                if best_model_y is not None:
                    X_all = np.concatenate((np.concatenate([m.get_X() for m in self.models]), best_model_y.get_X()))
                    y_all = np.concatenate((np.concatenate([m.get_y() for m in self.models]), best_model_y.get_y()))

                    unique_Xy, count = np.unique(np.concatenate((X_all, y_all), axis=1), axis=0, return_counts=True)

                    X_ext = unique_Xy[count == 1, : X_all.shape[1]]
                    y_ext = unique_Xy[count == 1, X_all.shape[1] :]

                    best_model_y.add_EV(x, y, np.concatenate((x_min, x_max, X_ext), axis=0), np.concatenate((y_min, y_max, y_ext), axis=0))
                else:
                    X_ext = np.concatenate([m.get_X() for m in self.models])
                    y_ext = np.concatenate([m.get_y() for m in self.models])

                    self.models.append(self.Cluster(self.tau, self.sigma))
                    self.models[-1].add_EV(x, y, np.concatenate((x_min, x_max, X_ext), axis=0), np.concatenate((y_min, y_max, y_ext), axis=0))               

        # Calculating statistics for a step k
        self.number_of_rules.append(len(self.models))        

def eevm(X_min, X, X_max, y_min, y, y_max, plot=-1):
    model = eEVM()

    predictions = np.zeros((y.shape[0], 1))

    start_time = time()

    for i in tqdm(range(y.shape[0])):
        predictions[i, 0] = model.predict(X[i, :].reshape(1, -1))        
        model.train(X_min[i, :].reshape(1, -1), X[i, :].reshape(1, -1), X_max[i, :].reshape(1, -1), y_min[i].reshape(1, -1), y[i].reshape(1, -1), y_max[i].reshape(1, -1))

        if plot != -1:
            if (i % plot) == 0:
                model.plot()

    return (predictions, model.number_of_rules[-1], time() - start_time, model.number_of_rules) 