# Least_SRMTL
# Sparse Structure-Regularized Learning with Least Squares Loss.
# Adapted from the MALSAR package by Amanda O. C. Ayres
# May 2020

# OBJECTIVE
# argmin_W { sum_i^t (0.5 * norm (Y{i} - X{i}' * W(:, i))^2)
#             + rho1 * norm(W*R, 'fro')^2 + rho2 * \|W\|_1}

# R encodes the structure relationship
# 1)Structure order is given by using [s12 -s12 0 ...; 0 s23 -s23 ...; ...]
# 2)Ridge penalty term by setting: R = eye(t)
# 3)All related regularized: R = eye (t) - ones (t) / t

# INPUT
#    X: {n * d} * t - input matrix
#    Y: {n * 1} * t - output matrix
#    R: regularization structure
#    rho1: structure regularization parameter
#    rho2: sparsity controlling parameter

# OUTPUT
#    W: model: d * t
#    funcVal: function value vector.

# RELATED PAPERS

# [1] Evgeniou, T. and Pontil, M. Regularized multi-task learning, KDD 2004
# [2] Zhou, J. Technical Report. http://www.public.asu.edu/~jzhou29/Software/SRMTL/CrisisEventProjectReport.pdf

import numpy as np

class Least_SRMTL(object):
    # Default values
    DEFAULT_MAX_ITERATION = 1000
    DEFAULT_TOLERANCE     = 1e-4
    DEFAULT_TERMINATION_COND = 1

    def __init__(self, rho_1=0, rho_2=0, rho_3=0):
        self.rho_1 = rho_1
        self.rho_2 = rho_2
        self.rho_3 = rho_3   
        self.funcVal = None
        self.W = None
    
    def funVal_eval(self, X, Y, W):
        funcVal = 0

        for i in range(self.t):
            funcVal = funcVal + 0.5 * np.linalg.norm(Y[i] - X[i].T @ W[:, i].reshape(-1, 1)) ** 2

        if self.R is None:
            return funcVal + self.rho_3 * np.linalg.norm(W, 'fro') ** 2
        return funcVal + self.rho_1 * np.linalg.norm(W @ self.R, 'fro') ** 2 + self.rho_3 * np.linalg.norm(W, 'fro') ** 2

    def gradVal_eval(self, X, XY, W):
        grad_W = np.zeros((X[0].shape[0], self.t))

        for t_ii in range(self.t):
            XWi = X[t_ii].T @ W[:,t_ii]
            XTXWi = X[t_ii] @ XWi
            grad_W[:, t_ii] = XTXWi - XY[t_ii].reshape(-1)

        if self.RRt is None:
            return grad_W + self.rho_3 * 2 * W
        return grad_W + self.rho_1 * 2 *  W @ self.RRt + self.rho_3 * 2 * W

    # Calculates argmin_z = \|z-v\|_2^2 + beta \|z\|_1 
    # z: solution, l1_comp_val: value of l1 component (\|z\|_1)
    @staticmethod
    def l1_projection(v, beta):        
        z = np.zeros(v.shape)
        vp = v - beta / 2
        z[np.where(v > beta / 2)]  = vp[np.where(v > beta / 2)]
        vn = v + beta / 2
        z[np.where(v < - beta / 2)] = vn[np.where(v < - beta / 2)]
        
        l1_comp_val = np.sum(np.abs(z))

        return [z, l1_comp_val]

    def multi_transpose(self, X):
        for i in range(self.t):
            X[i] = np.insert(X[i], 0, 1, axis=1).T
        
        return X

    def set_RRt(self, R):
        # precomputation        
        if R is None:
            self.RRt = None
        else:
            self.RRt = R @ R.T
        self.R = R

    def train(self, X, Y, init_theta=2):
        self.funcVal = list()
        X = X.copy()

        if init_theta == 1:
            W0 = self.W
        else:
            self.t = len(X)
            W0 = np.zeros((X[0].shape[1] + 1, self.t))
        
        X = self.multi_transpose(X)
        XY = list()

        for t_idx in range(self.t):
            XY.append(X[t_idx] @ Y[t_idx])
        
        # this flag tests whether the gradient step only changes a little
        bFlag = 0 

        Wz= W0
        Wz_old = W0

        t = 1
        t_old = 0


        gamma = 1
        gamma_inc = 2

        for iter_ in range(Least_SRMTL.DEFAULT_MAX_ITERATION):
            alpha = (t_old - 1) / t
            Ws = (1 + alpha) * Wz - alpha * Wz_old

            # compute the function value and gradients of the search point
            gWs = self.gradVal_eval(X, XY, Ws)
            Fs = self.funVal_eval(X, Y, Ws)

            while True:
                [Wzp, l1c_wzp] = self.l1_projection(Ws - gWs / gamma, 2 * self.rho_2 / gamma)
                Fzp = self.funVal_eval(X, Y, Wzp)
                
                delta_Wzp = Wzp - Ws
                r_sum = np.linalg.norm(delta_Wzp, ord='fro') ** 2

                Fzp_gamma = Fs + np.trace(delta_Wzp.T @ gWs) + gamma / 2 * np.linalg.norm(delta_Wzp, 'fro') ** 2
                
                if r_sum <= 1e-20:
                    # this shows that the gradient step makes little improvement
                    bFlag = 1  
                    break
                
                if Fzp <= Fzp_gamma:
                    break
                else:
                    gamma = gamma * gamma_inc
            
            Wz_old = Wz
            Wz = Wzp
            
            self.funcVal.append(Fzp + self.rho_2 * l1c_wzp)

            if bFlag:
                # The program terminates as the gradient step changes the solution very small
                break

            # test stop condition
            if iter_ >= 2:
                if abs(self.funcVal[-1] - self.funcVal[-2]) <= Least_SRMTL.DEFAULT_TOLERANCE * self.funcVal[-2]:
                    break

            t_old = t
            t = 0.5 * (1 + (1 + 4 * t ** 2) ** 0.5)
        
        self.W = Wzp

        W_return = list(self.W.T)
        return [w.reshape(1, -1) for w in W_return]