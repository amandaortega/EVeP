"""
    Author: Amanda Ortega de Castro Ayres
    Created in: September 19, 2019
    Python version: 3.6
"""

import libmr
from matplotlib import pyplot, cm
from matplotlib.patches import Circle
from mpl_toolkits.mplot3d import Axes3D, art3d
import numpy as np
import numpy.matlib
import sklearn.metrics

class eEVM(object):

    """
    evolving Extreme Value Machine
    Ruled-based predictor with EVM at the definition of the antecedent of the rules.    
    1. Create a new instance and provide the model parameters;
    2. Call the predict(x) method to make predictions based on the given input;
    3. Call the train(x, y) method to evolve the model based on the new input-output pair.
    """

    # Model initialization
    def __init__(self, sigma=0.5, tau=75, refresh_rate=50, window_size=np.Inf):        
        # Setting EVM algorithm parameters
        self.sigma = sigma
        self.tau = tau
        self.refresh_rate = refresh_rate
        self.window_size = window_size    
        self.last_cluster_id = -1

        self.mr_x = list()
        self.mr_y = list()
        self.x0 = list()
        self.y0 = list()
        self.X = list()
        self.y = list()
        self.step = list()
        self.last_update = list()
        self.theta = list()
        self.cluster = list()

    # Initialization of a new instance of EV.
    def add_EV(self, x0, y0, step, cluster, X=None, y=None, step_samples=None, X_ext=None, y_ext=None):
        self.mr_x.append(libmr.MR())
        self.mr_y.append(libmr.MR())
        self.x0.append(x0)
        self.y0.append(y0)
        self.X.append(x0)
        self.y.append(y0)
        self.step.append(step)
        self.last_update.append(np.max(step))
        self.cluster.append(cluster)
        self.theta.append(np.zeros_like(x0))

        if X_ext is not None:
            self.fit(len(self.cluster) - 1, X_ext, y_ext)

        if X is not None:
            self.add_sample_to_EV(len(self.x0) - 1, X, y, step_samples)
        else:
            # coefficients of the consequent part        
            self.theta[-1] = np.insert(self.theta[-1], 0, y0, axis=1).T

    # Add the sample(s) (X, y) as covered by the extreme vector. Remove repeated points.
    def add_sample_to_EV(self, index, X, y, step):
        self.X[index] = np.concatenate((self.X[index], X))
        self.y[index] = np.concatenate((self.y[index], y))
        self.step[index] = np.concatenate((self.step[index], step))          

        if self.X[index].shape[0] > self.window_size:
            indexes = np.argsort(-self.step[index].reshape(-1))

            self.X[index] = self.X[index][indexes[: self.window_size], :]
            self.y[index] = self.y[index][indexes[: self.window_size]]
            self.step[index] = self.step[index][indexes[: self.window_size]]
        
        self.x0[index] = np.average(self.X[index], axis=0).reshape(1, -1)
        self.y0[index] = np.average(self.y[index], axis=0).reshape(1, -1)

        (X_ext, y_ext) = self.get_external_samples(index)
        if X_ext.size > 0:
            self.fit(index, X_ext, y_ext)

        self.last_update[index] = np.max(self.step[index])
        self.theta[index] = np.linalg.lstsq(np.insert(self.X[index], 0, 1, axis=1), self.y[index])[0]    

    def delete_from_list(self, list_, indexes):
        for i in sorted(indexes, reverse=True):
            del list_[i]
        
        return list_

    # Calculate the firing degree of the sample to the psi curve
    def firing_degree(self, index, x=None, y=None):
        if y is None:
            return self.mr_x[index].w_score_vector(sklearn.metrics.pairwise.pairwise_distances(self.x0[index], x).reshape(-1))
        elif x is None:
            return self.mr_y[index].w_score_vector(sklearn.metrics.pairwise.pairwise_distances(self.y0[index], y).reshape(-1))
        else:
            return np.minimum(self.mr_x[index].w_score_vector(sklearn.metrics.pairwise.pairwise_distances(self.x0[index], x).reshape(-1)), self.mr_y[index].w_score_vector(sklearn.metrics.pairwise.pairwise_distances(self.y0[index], y).reshape(-1)))

    # Fit the psi curve of the EVs according to the external samples 
    def fit(self, index, X_ext, y_ext):
        self.fit_x(index, sklearn.metrics.pairwise.pairwise_distances(self.x0[index], X_ext)[0])
        self.fit_y(index, sklearn.metrics.pairwise.pairwise_distances(self.y0[index], y_ext)[0])

    # Fit the psi curve to the extreme values with distance D to the center of the EV
    def fit_x(self, index, D):
        self.mr_x[index].fit_low(1/2 * D, min(D.shape[0], self.tau))   

    # Fit the psi curve to the extreme values with distance D to the center of the EV
    def fit_y(self, index, D):
        self.mr_y[index].fit_low(1/2 * D, min(D.shape[0], self.tau))                                                  

    # Get the distance from the origin of the EV which has the given probability to belong to the curve
    def get_distance(self, index, percentage):
        return self.mr_x[index].inv(percentage)

    # Obtain the samples that not belong to the cluster given by parameter but are part of the other clusters of the system
    def get_external_samples(self, index=None):
        if index is None:
            X = np.concatenate(self.X)
            y = np.concatenate(self.y)
        else:
            if self.get_number_of_EVs() > 1:
                X = np.concatenate(self.X[:index] + self.X[index + 1 :])
                y = np.concatenate(self.y[:index] + self.y[index + 1 :])
            else:
                X = np.array([])
                y = np.array([])                

        return (X, y)

    def get_internal_samples(self, cluster):
        indexes = np.where(np.array(self.cluster) == cluster)
        return (np.concatenate(list(np.array(self.X)[indexes])), np.concatenate(list(np.array(self.y)[indexes])), np.concatenate(list(np.array(self.step)[indexes])))

    # Return the number of clusters existing in the model
    def get_number_of_clusters(self):        
        return len(np.unique(self.cluster))

    # Return the total number of EVs existing in the model
    def get_number_of_EVs(self):
        return len(self.mr_x)                

    def get_step(self, cluster):
        return np.concatenate(list(np.array(self.step)[np.where(np.array(self.cluster) == cluster)]))

    def get_X(self, cluster):
        return np.concatenate(list(np.array(self.X)[np.where(np.array(self.cluster) == cluster)]))

    def get_y(self, cluster):
        return np.concatenate(list(np.array(self.y)[np.where(np.array(self.cluster) == cluster)]))

    # Merge two EVs of different clusters whenever the origin of one is inside the sigma probability of inclusion of the psi curve of the other
    def merge(self):
        self.sort_EVs()
        index = 0

        while index < len(self.mr_x):
            if index + 1 < len(self.mr_x):
                x0 = np.concatenate(self.x0[index + 1 : ])
                y0 = np.concatenate(self.y0[index + 1 : ])

                S_index = self.firing_degree(index, x0, y0)
                index_to_merge = np.where(S_index > self.sigma)[0] + index + 1

                for i in reversed(range(len(index_to_merge))):
                    self.add_sample_to_EV(index, self.X[index_to_merge[i]], self.y[index_to_merge[i]], self.step[index_to_merge[i]])
                    self.remove_EV([index_to_merge[i]])
            
            index = index + 1

    # Plot the granules that form the antecedent part of the rules
    def plot(self, name_figure):
        fig = pyplot.figure()
        ax = fig.add_subplot(111, projection='3d')
        z_bottom = -0.3
        ax.set_zticklabels("")        

        colors = cm.get_cmap('tab20', len(np.unique(np.array(self.cluster))))

        for i in range(self.get_number_of_EVs()):
            self.plot_EV(i, ax, '.', colors(self.cluster[i]), z_bottom)
        
        # Save figure
        fig.savefig(name_figure)

        # Close plot
        pyplot.close(fig)

    # Plot the probability of sample inclusion (psi-model) together with the samples associated with the EV
    def plot_EV(self, index, ax, marker, color, z_bottom):
        # Plot the input samples in the XY plan
        ax.scatter(self.X[index][:, 0], self.X[index][:, 1], z_bottom * np.ones((self.X[index].shape[0], 1)), marker=marker, color=color)

        # Plot the radius for which there is a probability sigma to belong to the EV
        radius = self.get_distance(index, self.sigma)
        p = Circle((self.x0[index][0, 0], self.x0[index][0, 1]), radius, fill=False, color=color)
        ax.add_patch(p)
        art3d.pathpatch_2d_to_3d(p, z=z_bottom, zdir="z")

        # Plot the psi curve of the EV
        r = np.linspace(0, self.get_distance(index, 0.05), 100)
        theta = np.linspace(0, 2 * np.pi, 145)    
        radius_matrix, theta_matrix = np.meshgrid(r,theta)            
        X = self.x0[index][0, 0] + radius_matrix * np.cos(theta_matrix)
        Y = self.x0[index][0, 1] + radius_matrix * np.sin(theta_matrix)
        points = np.array([np.array([X, Y])[0, :, :].reshape(-1), np.array([X, Y])[1, :, :].reshape(-1)]).T
        Z = self.firing_degree(index, points)
        ax.plot_surface(X, Y, Z.reshape((X.shape[0], X.shape[1])), antialiased=False, cmap=cm.coolwarm, alpha=0.1)

    # Predict the output given the input sample x
    def predict(self, x):
        # Checking for system prior knowledge
        if len(self.mr_x) == 0:
            return np.mean(x)

        num = 0
        den = 0

        for i in range(len(self.mr_x)):
            p = self.predict_EV(i, x)

            num = num + self.firing_degree(i, x, p) * p
            den = den + self.firing_degree(i, x, p)

        if den == 0:
            return np.mean(x)

        return num / den

    # Predict the local output of x based on the linear regression of the samples stored at the EV
    def predict_EV(self, index, x):
        return np.insert(x, 0, 1).reshape(1, -1) @ self.theta[index]        

    # Remove all the the EVs belonging to the cluster informed by parameter
    def remove_cluster(self, cluster):
        self.remove_EV(list(np.where(np.array(self.cluster) == cluster)[0]))

    # Remove the EV whose index was informed by parameter
    def remove_EV(self, index):
        self.mr_x = self.delete_from_list(self.mr_x, index)
        self.mr_y = self.delete_from_list(self.mr_y, index)
        self.x0 = self.delete_from_list(self.x0, index)
        self.y0 = self.delete_from_list(self.y0, index)
        self.X = self.delete_from_list(self.X, index)
        self.y = self.delete_from_list(self.y, index)
        self.step = self.delete_from_list(self.step, index)
        self.last_update = self.delete_from_list(self.last_update, index)
        self.cluster = self.delete_from_list(self.cluster, index)
        self.theta = self.delete_from_list(self.theta, index)

    # Remove the EVs that didn't have any update in the last threshold steps
    def remove_outdated_EVs(self, threshold):
        indexes_to_remove = list()

        for index in range(len(self.last_update)):
            if self.last_update[index] <= threshold:
                indexes_to_remove.append(index)

        if len(indexes_to_remove) > 0:
            self.remove_EV(indexes_to_remove)    

    # Sort the EVs according to the last update
    def sort_EVs(self):
        new_order = (-np.array(self.last_update)).argsort()

        self.mr_x = list(np.array(self.mr_x)[new_order])
        self.mr_y = list(np.array(self.mr_y)[new_order])
        self.x0 = list(np.array(self.x0)[new_order])
        self.y0 = list(np.array(self.y0)[new_order])
        self.X = list(np.array(self.X)[new_order])
        self.y = list(np.array(self.y)[new_order])
        self.step = list(np.array(self.step)[new_order])
        self.last_update = list(np.array(self.last_update)[new_order])
        self.cluster = list(np.array(self.cluster)[new_order])

    # Evolves the model (main method)
    def train(self, x, y, step):
        # empty antecedents
        if len(self.mr_x) == 0:
            self.add_EV(x, y, step, 0)
            self.last_cluster_id = 0
        else:
            best_EV = None
            best_EV_value = 0
            
            cluster = None
            best_EV_y_value = 0

            # check if it is possible to insert the sample in an existing model
            for index in range(len(self.mr_x)):
                tau = self.firing_degree(index, x, y)

                if tau > best_EV_value and tau > self.sigma:
                    best_EV = index
                    best_EV_value = tau
                else:
                    tau = self.firing_degree(index, y=y)

                    if tau > best_EV_y_value and tau > self.sigma:
                        cluster = self.cluster[index]
                        best_EV_y_value = tau            
            
            # Add the sample to an existing EV
            if best_EV is not None:
                self.add_sample_to_EV(best_EV, x, y, step)
                cluster = self.cluster[best_EV]
                index = best_EV
            # Create a new cluster
            elif cluster is None:
                self.last_cluster_id = self.last_cluster_id + 1
                cluster = self.last_cluster_id      
                index = self.get_number_of_EVs()

            # Create a new EV in the respective cluster
            if best_EV is None:
                (X_ext, y_ext) = self.get_external_samples()                
                self.add_EV(x, y, step, cluster, X_ext=X_ext, y_ext=y_ext)
            
            self.update_EVs(index)

        if step != 0 and (step % self.refresh_rate) == 0:      
            self.remove_outdated_EVs(step[0, 0] - self.refresh_rate)
            self.merge()

    # Update the psi curve of the EVs that do not belong to the model_selected
    def update_EVs(self, index):
        for i in range(len(self.mr_x)):
            if i != index:
                (X_ext, y_ext) = self.get_external_samples(i)

                if X_ext.shape[0] > 0:
                    self.fit(i, X_ext, y_ext)