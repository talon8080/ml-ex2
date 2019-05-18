import numpy as np
from sklearn.utils import shuffle
from scipy import stats
import operator
import sys
from abc import ABC, abstractmethod
import random


class Model(ABC):
    '''
    Model Class interface
    '''

    def __init__(self,epochs=10):
        self.w = None
        self.epochs = epochs
        super(Model, self).__init__()

    @abstractmethod
    def fit(self,X,Y):
        pass

    def predict(self,x_test,y_test):
        e_perceptron = 0
        c_perceptron = 0
        labels = [-1] * len(x_test)
        for t in range(0, len(x_test)):
            labels[t] = np.argmax(np.dot(self.w, x_test[t]))
            if labels[t] != y_test[t]:
                e_perceptron = e_perceptron + 1
            else:
                c_perceptron = c_perceptron + 1
        print("error: {}".format(float(e_perceptron)/len(x_test)))
        print("correct: {}".format(float(c_perceptron)/len(x_test)))
        return labels



class Perceptron(Model):

    def fit(self,X,Y):
        #w = np.random.uniform(-0.5,0.5,(3,X.shape[1]))
        w = np.array([[-0.37370176,0.02309853,-0.02886409,-0.57941457,0.35742136,-0.19498365,-0.39270688,-0.4430768,-0.1098553,0.4713786,0.10372131],
        [-0.05160724,0.43913619,0.1081243,-0.75047572,0.23008437,-0.11137963,0.02179128,0.17009974,0.42444943,-0.1830452,-0.06628518],
        [-0.26307821,0.64616801,0.39153138,0.33317525,-0.6241195,-0.40444592,0.35822757,0.20053011,0.33321939,0.44802492,-0.24912969]])
        eta = 0.1805152982621216
        #eta = np.random.uniform(0.01,0.3)
        accuracy_list = np.zeros(self.epochs)
        for epoch in range(self.epochs):
            x_train, y_train, x_valid, y_valid = validation_set(X, Y)
            # Need to shuffle on X and Y 
            for x,y in zip(x_train,y_train):
                y_hat = np.argmax(np.dot(w,x))
                if y_hat != y:
                    w[int(y),:] = w[int(y),:] + eta * x
                    w[int(y_hat),:] = w[int(y_hat),:] - eta * x

            accuracy_list[epoch] = pred_valid(x_valid, y_valid, w)
        accuracy_model = accuracy_list.mean()
        print("eta: {}\n accuracy: {}".format(eta, accuracy_model))
        self.w = w
        return w




class PA(Model):
    def fit(self,X,Y):
        #w = np.random.uniform(-0.5,0.5,(3,X.shape[1]))
        w = np.array([[-0.78513436,0.57848318,0.21929112,-0.71731913,1.05820117,0.85819699,-1.49767045,-0.124864,0.02344565,-0.34982529,0.04340942],
        [-0.02670186,-0.21029281,-0.00643031,-1.02291055,1.40160315,0.078589,0.60966336,-0.31403649,0.39471659,0.24901959,0.11034403],
        [0.67687675,-0.20768354,0.32662767,2.54970554,-2.40675052,-1.36631277,1.12136024,0.50403388,0.04946138,-0.37992296,-0.18701959]])
        accuracy_list = np.zeros(self.epochs)
        for epoch in range(self.epochs):
            x_train, y_train, x_valid, y_valid = validation_set(X, Y)
            for x,y in zip(x_train,y_train):
                y_hat = np.argmax(np.dot(w,x))
                loss = max([0.0, 1.0 - np.dot(w[int(y),:], x) + np.dot(w[int(y_hat),:], x)])
                if loss > 0.0:
                    # if not np.all(x == 0.0):
                    #     tau = loss / float(2 * np.dot(x,x))
                    # else:
                    #     tau = loss / X.shape[0]
                    tau = 0.0509912
                    w[int(y),:] = w[int(y),:] + tau * x
                    w[int(y_hat),:] = w[int(y_hat),:] - tau * x

            accuracy_list[epoch] = pred_valid(x_valid, y_valid, w)
        accuracy_model = accuracy_list.mean()
        print("tau: {}\n accuracy: {}".format(tau, accuracy_model))
        self.w = w
        return w
        





class SVM(Model):

    def fit(self,X,Y):
        #w = np.random.uniform(-0.5,0.5,(3,X.shape[1]))
        w = np.array([[-7.41114338e-02,-3.24392462e-01,-3.61195958e-01,-1.99673854e-01,7.52323007e-02,-1.45902499e-01,-6.31243562e-01,3.80050868e-04,3.80050868e-04,3.80050868e-04,3.80050868e-04],
        [8.33572613e-02,1.05233292e-01,1.23141150e-01,8.26881796e-02,3.01419212e-01,1.63216870e-01,-1.63629064e-01,-4.99656050e-05,-4.99656050e-05,-4.99656050e-05,-4.99656050e-05],
        [-6.94694745e-02,1.81936871e-01,2.31649111e-01,1.12357283e-01,-5.02994218e-01,-4.53156554e-02,9.01373556e-01,-3.70156494e-04,-3.70156494e-04,-3.70156494e-04,-3.70156494e-04]
        ])
        alpha = 0.1044000343
        accuracy_list = np.zeros(self.epochs)
        eta = 0.280086
        for epoch in range(self.epochs):
            x_train, y_train, x_valid, y_valid = validation_set(X, Y)
            for x, y in zip(x_train, y_train):
                y_hat = np.argmax(np.dot(w,x))
                if y_hat != y:
                    w[int(y), :] = (1 - eta * alpha) * w[int(y), :] + eta * x
                    w[int(y_hat), :] = (1 - eta * alpha) * w[int(y_hat), :] - eta * x
                for i in range(w.shape[0]):
                    if i != int(y) and i != int(y_hat):
                        w[i, :] = (1 - eta * alpha) * w[i, :]
            accuracy_list[epoch] = pred_valid(x_valid, y_valid, w)
        accuracy_model = accuracy_list.mean()
        print("alpha: {}\n accuracy: {}\n eta:{}".format(alpha, accuracy_model, eta))
        self.w = w
        return w






def validation_set(x_train, y_train):
    
    permutation = np.random.permutation(x_train.shape[0])
    x_train, y_train = x_train[permutation], y_train[permutation]
    x_valid = x_train[-int(np.round(0.2*len(x_train))):]
    x_train = x_train[:int(np.round(0.8*len(x_train)))]
    y_valid = y_train[-int(np.round(0.2*len(y_train))):]
    y_train = y_train[:int(np.round(0.8*len(y_train)))]

    return x_train, y_train, x_valid, y_valid


def pred_valid(x_valid, y_valid, w):

    c_perceptron = 0
    for t in range(0,len(x_valid)):
        y_hat = np.argmax(np.dot(w,x_valid[t]))
        if y_hat == y_valid[t]:
            c_perceptron = c_perceptron + 1
    return float(c_perceptron)/len(x_valid)


def load_ds(train_x_file_name,train_y_file_name):
    x_train = np.loadtxt(train_x_file_name, usecols=(1,2,3,4,5,6,7), dtype=np.float32, delimiter=',')
    y_train = np.loadtxt(train_y_file_name)
    # x_test = np.loadtxt(test_file_name, usecols=(1,2,3,4,5,6,7), dtype=np.float32, delimiter=',')
    permutation = np.random.permutation(x_train.shape[0])
    x_train, y_train = x_train[permutation], y_train[permutation]
    x_test = x_train[-int(np.round(0.3*len(x_train))):]
    x_train = x_train[:int(np.round(0.7*len(x_train)))]
    y_test = y_train[-int(np.round(0.3*len(y_train))):]
    y_train = y_train[:int(np.round(0.7*len(y_train)))]

    return x_train, y_train, x_test, y_test


def normalizeData(X):
    m, d = X.shape
    for i in range(d):
        cmean = X[:,i].mean()
        cstd = X[:,i].std()
        if cstd == 0:
            X[:,i] = 1.0 / float(m)
        else:
            X[:,i] = (X[:,i] - cmean) / cstd
    return X


def prepareData(X):
    
    X_dummis = list()
    for i,line in enumerate(X):
        if line[0] == 'M':
            appendcol = [1.,0.,0.,1.]        
        elif line[0] == 'F':
            appendcol = [0.,1.,0.,1.]        
        else:
            appendcol = [0.,0.,1.,1.]
        X_dummis.append(np.append(X[i],appendcol))
    X = np.array(X_dummis).astype('float32')
    return X

def min_max_scaling(X):
    n_samples = X.shape[0]
    n_features = X.shape[1]

    for i in range(n_features):
        V = X[:,i]
        V_min = V.min()
        V_max = V.max()

        if V_max != V_min:
            V_tag = np.true_divide(V-V_min,V_max-V_min)
            V_tag_min = V_tag.min()
            V_tag_max = V_tag.max()
            V_tag *= (V_tag_max-V_tag_min)
            V_tag+= V_tag_min
            X[:,i] = V_tag
        else: # V_max == V_min:
            X[:, i] = np.true_divide(1,n_samples)

    return X

if __name__ == "__main__":

    train_x_file_name = sys.argv[1]
    train_y_file_name = sys.argv[2]
    # test_file_name = sys.argv[3]

    
    x_train, y_train, x_test, y_test = load_ds(train_x_file_name, train_y_file_name)

    x_train = prepareData(x_train)
    x_test = prepareData(x_test)

    # y_test = np.loadtxt("test_y.txt")
    
    x_train = normalizeData(x_train)
    x_test = normalizeData(x_test)

    model_names = ['Perceptron', 'SVM', 'PA']
    predicts = {}
    for model_name in model_names:
        random.seed()
        model = eval(model_name)(epochs = 10)
        print("\n")
        w = model.fit(x_train, y_train)
        predicts[model_name.lower()] = model.predict(x_test,y_test)
        print("\n")

    # for i in range(len(x_test)):
    #     for model_name in model_names:
    #         if model_name.lower == 'svm':
    #             suffix = ''
    #         else:
    #             suffix = ', '
    #         print(model_name.lower() + ": " + str(predicts[model_name.lower()][i]) + suffix,end='')
    #     print()
     

    