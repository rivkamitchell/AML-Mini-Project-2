import numpy as np
#no tested 

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
        for i in range(self.y_range):
            y_list+=[[]]

        #separates into different y-value list, so size y_range list. list of list

        for row in self.data:
           y_list[row[-1]]+=[row]

        for p in range(y_range):
            tot=len(y_list[p])
            vector[p]=tot/self.num_exp #prob of y=p
            if tot!=0:
                for i in range(slef.num_features):
                    num_of_i=0
                    for row in y_list[p]:
                        if row[i]==1:
                            num_of_i+=1
                    theta[i, p]=num_of_i/tot #prob of ith =1 knowing that y=p
            else:
                #??????
                print("problem")
        return (vector, theta)


    def predict(self, vect):
        max_prob=0
        max_class=0
        for p in range(self.y_range):
            prob=self.theta_vect[p]
            for i in range(self.num_features):
                if vect[i]==1:
                    prob*=self.theta_mat[i, p]
                else:
                    prob*=(1-self.theta_mat[i, p])
            if prob>max_prob:
                max_prob=prob
                max_class=p
        return max_class



    def accuracy(self, vali):
        tr = 0 #number of right predictions
        tw = 0 #number of wrong predictions
        self.validation=vali

        for i in range(len(self.validation)):
            if self.results[i]==self.predict(self.features[i]) :
                tr+=1
            else: 
                tw+=1

        return (tr/(tw+tr))

    def k_fold_crossing(self, k):
        size=int(self.num_exp/k)
        acc=0
        for i in range(k):
            temp1=i*size
            temp2=(i+1)*size
            train=np.vstack((self.data[:temp1], self.data[temp2:]))
            vali=self.data[temp1:temp2]
            model=NAIVE(train, self.x_range, self.y_range)
            acc+=model.accuracy(vali)
        return (acc/k)
