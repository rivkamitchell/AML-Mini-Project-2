# Not tested

import numpy as np
import Preprocessing 

class NAIVE:

    def __init__(self, X, x_range, y_range): #x_range should be a discrete set(binary for now) as well as y_range
        X=np.array(X)
        self.data = X
        self.validation=X
        (self.num_exp, self.num_features)=X.shape
        self.features = X[:,:len(X[0])-1]
        self.results=X[:, -1]
        self.x_range=2
        self.y_range=y_range
        (self.theta_vect, self.theta_mat)=self.fit()

    def fit(self):
        #calculates the theta matrix 
        theta=np.zeros(self.num_features, self.y_range)
        vector=np.zeros(self.y_range) #prob of y=p

        y_list=[]
"NAIVE" 87L, 2579C

