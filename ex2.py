import numpy as np
from scipy import stats
import operator
import sys
from sklearn.utils import shuffle as ss
from abc import ABC, abstractmethod
from random import randrange, shuffle, seed

class Model(ABC):
    '''
    Model Abstract Class
    '''

    def __init__(self,epochs=10):
        self.w = None
        self.model_name = "model"
        self.epochs = epochs
        super(Model, self).__init__()

    @abstractmethod
    def fit(self,X,Y,X_test,Y_test):
        pass

    def predict(self,x_test,y_test=None,to_print=False):
        error_count = success_count = 0
        res = [-1] * len(x_test)
        for t in range(0, len(x_test)):
            res[t] = np.argmax(np.dot(self.w, x_test[t]))

            if y_test is not None:
                if res[t] != y_test[t]:
                    error_count+=1
                else:
                    success_count+=1

        if to_print and y_test is not None:
            print(self.model_name + ":")
            print("accuracy: {}".format(float(success_count)/len(x_test)),end=' ')
            print("error: {}".format(float(error_count)/len(x_test)))
        return res


class Perceptron(Model):

    def __init__(self, *args, **kwargs):
        super(Perceptron, self).__init__(*args, **kwargs)
        self.model_name = 'Perceptron'

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

    def __init__(self, *args, **kwargs):
        super(SVM, self).__init__(*args, **kwargs)
        self.model_name = 'SVM'

    def fit(self,X,Y,X_test,Y_test,w=None):
        x_train = X.copy()
        y_train = Y.copy()
        x_test = X_test.copy()
        y_test = Y_test.copy()
        best_acc = -1
        alpha = np.random.random()
        eta = np.random.uniform(0.001, 0.5)

        if w is None:
            w = np.zeros((3,x_cv_train.shape[1]))

        svm_w_list = []
        #x_train, x_test, y_train, y_test = split_train_test_with_prop_random(X, Y)
        for epoch in range(self.epochs):
            shuffle(x_train)
            shuffle(y_train)
            shuffle(x_test)
            shuffle(y_test)
            for x, y in zip(x_train, y_train):
                y_hat = np.argmax(np.dot(w, x))
                if y_hat != y:
                    w[int(y), :] = (1 - eta * alpha) * w[int(y), :] + eta * x
                    w[int(y_hat), :] = (1 - eta * alpha) * w[int(y_hat), :] - eta * x
                for i in range(w.shape[0]):
                    if i != y and i != y_hat:
                        w[i, :] = (1 - eta * alpha) * w[i, :]

            svm_w_list.append(w.copy())


        idx , score = validate(svm_w_list,x_test,y_test)
        print(np.true_divide(score,y_test.shape[0]))
        if score > best_acc:
            best_acc = score
            self.w = svm_w_list[idx]

        # self.w = w

        return self.w


def prepareData(train_x_file_name,train_y_file_name,test_file_name):
    arr = np.loadtxt(train_x_file_name, usecols=(1,2,3,4,5,6,7), dtype=np.float32, delimiter=',')
    df = list()
    with open(train_x_file_name,'r') as file:
        for i,line in enumerate(file):
            if line[0] == 'M':
                appendcol = [1.,0.,0.]        
            elif line[0] == 'F':
                appendcol = [0.,1.,0.]        
            elif line[0] == 'I':
                appendcol = [0.,0.,1.]
            df.append(np.append(arr[i],appendcol))
    x_train = np.array(df).astype('float32')
    y_train = np.loadtxt(train_y_file_name)
    return x_train, y_train

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

def cross_validation_split(dataset, folds=3):
    dataset_split = list()
    dataset_copy = list(dataset)
    fold_size = int(len(dataset) / folds)
    for i in range(folds):
        fold = list()
        while len(fold) < fold_size:
            index = randrange(len(dataset_copy))
            fold.append(dataset_copy.pop(index))
        dataset_split.append(fold)
    return dataset_split


def pred_valid(x_valid, y_valid, w):

    c_perceptron = 0
    for t in range(0,len(x_valid)):
        y_hat = np.argmax(np.dot(w,x_valid[t]))
        if y_hat == y_valid[t]:
            c_perceptron = c_perceptron + 1
    return float(c_perceptron)/len(x_valid)


def load_ds(train_x_file_name,train_y_file_name,test_file_name):

    # reads x_train file and creates dummies
    X= np.loadtxt(train_x_file_name, dtype=str, delimiter=',')
    I_dummy = np.array(list(map(lambda x: 1 if x == 'I' else 0, X[:, 0]))).reshape(-1,1)
    F_dummy = np.array(list(map(lambda x: 1 if x == 'F' else 0, X[:, 0]))).reshape(-1,1)
    M_dummy = np.array(list(map(lambda x: 1 if x == 'M' else 0, X[:, 0]))).reshape(-1,1)
    x_train = np.concatenate([I_dummy, F_dummy, M_dummy, X[:, 1:].astype(float)], axis=1)

    # reads y_train file
    y_train = np.loadtxt(train_y_file_name,dtype=float).reshape(-1,1)

    # reads x_test file and creates dummies
    X_test = np.loadtxt(test_file_name, dtype=str, delimiter=',')
    I_dummy = np.array(list(map(lambda x: 1 if x == 'I' else 0, X_test[:, 0]))).reshape(-1, 1)
    F_dummy = np.array(list(map(lambda x: 1 if x == 'F' else 0, X_test[:, 0]))).reshape(-1, 1)
    M_dummy = np.array(list(map(lambda x: 1 if x == 'M' else 0, X_test[:, 0]))).reshape(-1, 1)
    x_test = np.concatenate([I_dummy, F_dummy, M_dummy, X_test[:, 1:].astype(float)], axis=1)

    # reads y_test file
    # todo remove before submission!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    y_test = np.loadtxt("test_y_30.txt", dtype=float).reshape(-1,1)

    return x_train, y_train, x_test, y_test

def split_train_test_with_prop_random(X, y, prop=.7):
    X = X.copy()
    index_random = np.arange(0, len(X))
    shuffle(index_random)
    index_train = index_random[0: int(prop * len(index_random))]
    index_test = index_random[int(prop * len(index_random)): len(index_random)]
    X_train = X.copy()[index_train]
    y_train = y.copy()[index_train]
    X_test = X.copy()[index_test]
    y_test = y.copy()[index_test]
    return X_train, X_test, y_train, y_test

def validate(w_list, x_valid, y_valid):
    w_dict = {}
    for i in range(len(w_list)):
        w = w_list[i]
        w_dict[i] = 0
        for t in range(0,len(x_valid)):
            y_hat = np.argmax(np.dot(w,x_valid[t]))
            if y_hat == y_valid[t]:
                w_dict[i] = w_dict[i] + 1
    return max(w_dict.items(), key=lambda x:x[1])


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

def create_folds(data,nfolds=10):

    folds = list()
    fold_size = int(data.shape[0] / nfolds)
    s = 0
    e = fold_size
    for i in range(1, nfolds + 1):
        if i == nfolds:
            fold = data[s:]
        else:
            fold = data[s:e]
        folds.append(fold)
        s = e
        e += fold_size
    return folds


if __name__ == "__main__":

    train_x_file_name = sys.argv[1]
    train_y_file_name = sys.argv[2]
    test_file_name = sys.argv[3]

    # Load data and convert categorical feature to dummies
    x_train, y_train, x_test, y_test = load_ds(train_x_file_name, train_y_file_name, test_file_name)

    # scale data using min max scaler
    x_train = min_max_scaling(x_train)
    x_test = min_max_scaling(x_test)

    # run models
    #model_names = ['Perceptron', 'SVM', 'PA']
    model_names = ['SVM']
    predicts = {}

    # perform cross validation
    for model_name in model_names:
        w = np.zeros((3, x_train.shape[1]))
        model = eval(model_name)(epochs=100)
        print("\n")
        nfolds = 8
        best_acc = -1
        folds = create_folds(np.concatenate([x_train,y_train],axis=1),nfolds)
        for i in range(nfolds):

            prev_w = w
            #train set
            train_merged = np.concatenate(folds[:i] + folds[i + 1:])
            x_cv_train = train_merged[:,:x_train.shape[1]]
            y_cv_train = train_merged[:,x_train.shape[1]]

            # test set
            x_cv_test = folds[i][:,:x_train.shape[1]]
            y_cv_test = folds[i][:,x_train.shape[1]]

            w = model.fit(x_cv_train, y_cv_train, x_cv_test, y_cv_test,w)

            Ws = [prev_w,w]
            idx, score = validate(Ws, x_cv_test, y_cv_test)

            if score > best_acc:
                best_acc = score
                w = Ws[idx]


        predicts[model_name.lower()] = model.predict(x_test,y_test,to_print=True)
        print("\n")

    # for i in range(len(x_test)):
    #     for model_name in model_names:
    #         if model_name.lower == 'svm':
    #             suffix = ''
    #         else:
    #             suffix = ', '
    #         print(model_name.lower() + ": " + str(predicts[model_name.lower()][i]) + suffix,end='')
    #     print()
     

    