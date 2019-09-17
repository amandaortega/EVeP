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
import numpy.matlib
import sklearn.metrics

class eEVM_RLS(object):

    """
    evolving Extreme Value Machine
    Ruled-based predictor with EVM at the definition of the antecedent of the rules.    
    1. Create a new instance and provide the model parameters;
    2. Call the predict(x) method to make predictions based on the given input;
    3. Call the train(x, y) method to evolve the model based on the new input-output pair.
    """

    class Cluster(object):
        """
        Cluster grouping near samples.
        """
        
        class EV(object):
            """
            Extreme vector.
            """

            P0 = 10**5

            # Initialization of a new instance of EV.
            def __init__(self, x0, y0, tau, step, window_size):
                self.mr_x = libmr.MR()
                self.mr_y = libmr.MR()
                self.mr_xy = libmr.MR()
                self.tau = tau
                self.x0 = x0
                self.y0 = y0
                self.X = x0
                self.y = y0
                self.step = step
                self.window_size = window_size
                self.last_update = np.max(step)

                # coefficients of the consequent part
                self.theta = np.zeros_like(x0)
                self.theta = np.insert(self.theta, 0, y0, axis=1).T

                self.P0 = y0[0,0]*self.P0
                self.P = self.P0 * np.eye(x0.shape[1] + 1)

            # Add the sample(s) (X, y) as covered by the extreme vector. Remove repeated points.
            def add_sample(self, X, y, step, theta=None):
                self.X = np.concatenate((self.X, X))                
                self.y = np.concatenate((self.y, y))
                self.step = np.concatenate((self.step, step))          

                if theta is None:
                    X = np.insert(X, 0, 1, axis=1).T
                    self.P = self.P @ (np.eye(X.shape[0]) - X @ X.T @ self.P / (1 + X.T @ self.P @ X))
                    self.theta = self.theta + (self.P @ X @ (y - X.T @ self.theta))
                else:
                    self.theta = (self.theta + theta) / 2                            

                if self.X.shape[0] > self.window_size:
                    indexes = np.argsort(-self.step.reshape(-1))

                    self.X = self.X[indexes[: self.window_size], :]
                    self.y = self.y[indexes[: self.window_size]]
                    self.step = self.step[indexes[: self.window_size]]
                
                self.last_update = np.max(self.step)

            # Calculate the firing degree of the sample to the psi curve
            def firing_degree(self, x=None, y=None):
                if y is None:
                    return self.mr_x.w_score_vector(sklearn.metrics.pairwise.pairwise_distances(self.x0, x).reshape(-1))
                elif x is None:
                    return self.mr_y.w_score_vector(sklearn.metrics.pairwise.pairwise_distances(self.y0, y).reshape(-1))
                else:
                    return self.mr_xy.w_score_vector(sklearn.metrics.pairwise.pairwise_distances(np.concatenate((self.x0, self.y0), axis=1), np.concatenate((x, y), axis=1)).reshape(-1))

            # Fit the psi curve of the EVs according to the external samples 
            def fit(self, X_ext, y_ext):
                self.fit_x(sklearn.metrics.pairwise.pairwise_distances(self.x0, X_ext)[0])
                self.fit_y(sklearn.metrics.pairwise.pairwise_distances(self.y0, y_ext)[0])
                self.fit_xy(sklearn.metrics.pairwise.pairwise_distances(np.concatenate((self.x0, self.y0), axis=1), np.concatenate((X_ext, y_ext), axis=1))[0])                

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
                Z = self.firing_degree(points)
                ax.plot_surface(X, Y, Z.reshape((X.shape[0], X.shape[1])), antialiased=False, cmap=cm.coolwarm, alpha=0.1)

            # Predict the local output of x based on the linear regression of the samples stored at the EV
            def predict(self, x):
                return np.insert(x, 0, 1).reshape(1, -1) @ self.theta

        def __init__(self, tau=75, sigma=0.5, window_size=50):
            self.EVs = list()
            self.tau = tau
            self.sigma = sigma
            self.window_size = window_size

        def add_EV(self, x0, y0, X_ext, y_ext, step):
            self.EVs.append(self.EV(x0, y0, self.tau, step, self.window_size))
            self.EVs[-1].fit(X_ext, y_ext)

        def firing_degrees(self, x, y):
            return np.array([ev.firing_degree(x, y) for ev in self.EVs])
        
        def get_samples(self):
            X = np.concatenate([ev.X for ev in self.EVs])
            y = np.concatenate([ev.y for ev in self.EVs])
            step = np.concatenate([ev.step for ev in self.EVs])

            return (X, y, step)
        
        def get_step(self):
            return np.concatenate([ev.step for ev in self.EVs])        

        def get_X(self):
            return np.concatenate([ev.X for ev in self.EVs])           

        def get_y(self):
            return np.concatenate([ev.y for ev in self.EVs])

        # Plot the probability of sample inclusion (psi-model) for each extreme value
        def plot(self, ax, marker, color, z_bottom):
            for ev in self.EVs:
                ev.plot(ax, marker, color, z_bottom, self.sigma)


        # Remove the EVs that didn't have any update in the last threshold steps
        def remove_outdated_EVs(self, threshold):
            index = 0

            while index < len(self.EVs):
                EV = self.EVs[index]

                if EV.last_update <= threshold:
                    self.EVs.remove(EV)
                else:
                    index = index + 1                

    # Model initialization
    def __init__(self, sigma=0.5, tau=75, refresh_rate=50, window_size=np.Inf):        
        # Setting local models
        self.models = list()

        # Setting EVM algorithm parameters
        self.sigma = sigma
        self.tau = tau
        self.refresh_rate = refresh_rate
        self.window_size = window_size

    # Return all the EVs and the respective model's index to which they belong
    def get_EVs(self):
        EVs = np.concatenate([m.EVs for m in self.models])
        how_many_EVs = [len(m.EVs) for m in self.models]

        EV_model_index = list()

        for i, how_many in enumerate(how_many_EVs):
            EV_model_index.extend([i] * how_many)

        return (EVs, EV_model_index)        

    # Obtain the EVs that not belong to the cluster given by parameter but are part of the other clusters of the system
    def get_external_EVs(self, cluster):
        return np.concatenate([m.EVs for m in self.models if m != cluster])

    # Obtain the samples that not belong to the cluster given by parameter but are part of the other clusters of the system
    def get_external_samples(self, cluster):
        X_all = np.concatenate((np.concatenate([m.get_X() for m in self.models]), cluster.get_X()))
        y_all = np.concatenate((np.concatenate([m.get_y() for m in self.models]), cluster.get_y()))

        unique_Xy, count = np.unique(np.concatenate((X_all, y_all), axis=1), axis=0, return_counts=True)

        X_ext = unique_Xy[count == 1, : X_all.shape[1]]
        y_ext = unique_Xy[count == 1, X_all.shape[1] :]        

        return (X_ext, y_ext)


    # Return the number of clusters existing in the model
    def get_number_of_clusters(self):
        return len(self.models)

    # Return the total number of EVs existing in the model
    def get_number_of_EVs(self):
        return np.concatenate([m.EVs for m in self.models]).shape[0]

    # Merge two EVs of different clusters whenever the origin of one is inside the sigma probability of inclusion of the psi curve of the other
    def merge(self):
        merged = False

        (EVs, EVs_model_index) = self.get_EVs()
        (EVs, EVs_model_index) = self.sort_EVs(EVs, EVs_model_index)    

        for index, EV in enumerate(EVs):
            if EVs[index + 1 :].size > 0:
                x0 = np.concatenate([other_EV.x0 for other_EV in EVs[index + 1 :]])
                y0 = np.concatenate([other_EV.y0 for other_EV in EVs[index + 1 :]])

                S_index = EV.firing_degree(x0, y0)
                index_to_merge = np.where(S_index > self.sigma)[0] + index + 1

                for i in reversed(range(len(index_to_merge))):
                    merged = True
                    EVs[index].add_sample(EVs[index_to_merge[i]].X, EVs[index_to_merge[i]].y, EVs[index_to_merge[i]].step, EVs[index_to_merge[i]].theta)
                    EVs = np.delete(EVs, index_to_merge[i])
                    del EVs_model_index[index_to_merge[i]]

        if merged:        
            actual_i = 0

            # remove the models that don't have any EV and update the EVs of those which remain
            for i in range(len(self.models)):
                if i not in EVs_model_index:
                    del self.models[actual_i]
                else:
                    self.models[actual_i].EVs = EVs[[j for j, x in enumerate(EVs_model_index) if x == i]].tolist()
                    actual_i = actual_i + 1                 

    # Plot the granules that form the antecedent part of the rules
    def plot(self, name_figure):
        fig = pyplot.figure()
        ax = fig.add_subplot(111, projection='3d')
        z_bottom = -0.3
        ax.set_zlim(bottom=z_bottom, top=1)
        ax.set_xlim(left=0, right=1)
        ax.set_ylim(bottom=0, top=1)
        ax.set_zticklabels("")        

        colors = cm.get_cmap('tab20', len(self.models))

        for i, m in enumerate(self.models):
            m.plot(ax, '.', colors(i), z_bottom)
        
        # Save figure
        fig.savefig(name_figure)

        # Close plot
        pyplot.close(fig)

    # Predict the output given the input sample x
    def predict(self, x):
        # Checking for system prior knowledge
        if len(self.models) == 0:
            return np.mean(x)

        num = 0
        den = 0

        for cluster in self.models:
            for EV in cluster.EVs:
                p = EV.predict(x)

                if EV.firing_degree(y=p) > self.sigma:
                    num = num + EV.firing_degree(x) * p
                    den = den + EV.firing_degree(x)

        if den == 0:
            return np.mean(x)

        return num / den

    def remove_outdated_EVs(self, step): 
        index = 0

        while index < len(self.models):
            cluster = self.models[index]
            cluster.remove_outdated_EVs(step - self.refresh_rate)

            # there is no more EV in the current cluster
            if len(cluster.EVs) == 0:
                self.models.remove(cluster)
            else:
                index = index + 1            

    # Sort the EVs based on the number of samples belonged to them
    def sort_EVs(self, EVs, EVs_model_index):
        how_many_samples = np.array([EV.X.shape[0] for EV in EVs])
        new_order = (-how_many_samples).argsort()
        return(EVs[new_order], np.array(EVs_model_index)[new_order].tolist())

    # Evolves the model (main method)
    def train(self, x_min, x, x_max, y_min, y, y_max, step):
        # empty antecedents
        if len(self.models) == 0:
            self.models.append(self.Cluster(self.tau, self.sigma, self.window_size))
            self.models[-1].add_EV(x, y, np.concatenate((x_min, x_max), axis=0), np.concatenate((y_min, y_max), axis=0), step)
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
                    best_model_y = m
                elif firing_degrees_y[best_index_y] > best_EV_y_value and firing_degrees_y[best_index_y] > self.sigma:
                    best_model_y = m
                    best_EV_y_value = firing_degrees_y[best_index_y]                    
            
            if best_EV is not None:
                best_EV.add_sample(x, y, step)
            else:
                if best_model_y is not None:
                    (X_ext, y_ext) = self.get_external_samples(best_model_y)
                    best_model_y.add_EV(x, y, np.concatenate((x_min, x_max, X_ext), axis=0), np.concatenate((y_min, y_max, y_ext), axis=0), step)                    
                else:
                    X_ext = np.concatenate([m.get_X() for m in self.models])
                    y_ext = np.concatenate([m.get_y() for m in self.models])

                    self.models.append(self.Cluster(self.tau, self.sigma, self.window_size))
                    self.models[-1].add_EV(x, y, np.concatenate((x_min, x_max, X_ext), axis=0), np.concatenate((y_min, y_max, y_ext), axis=0), step)               

                    best_model_y = self.models[-1]
            
            self.update_EVs(best_model_y)            

        if step != 0 and (step % self.refresh_rate) == 0:      
            self.remove_outdated_EVs(step[0, 0])      

            self.merge()

    # Update the psi curve of the EVs that do not belong to the model_selected
    def update_EVs(self, model_selected):
        for m in self.models:
            if m != model_selected:
                (X_ext, y_ext) = self.get_external_samples(m)

                if X_ext.shape[0] > 0:
                    for EV in m.EVs:       
                        EV.fit(X_ext, y_ext) 