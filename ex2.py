import numpy as np
from scipy import stats
import sys
from sklearn.utils import shuffle as ss
from abc import ABC, abstractmethod
from random import randrange, shuffle

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
    def fit(self,x_train,y_train):
        pass

    def predict(self,x_test,y_test,to_print=False):
        error_count = success_count = 0
        res = [-1] * len(x_test)
        for t in range(0, len(x_test)):
            res[t] = np.argmax(np.dot(self.w, x_test[t]))

            if res[t] != y_test[t]:
                error_count+=1
            else:
                success_count+=1

        if to_print:
            print(self.model_name + ":")
            print("accuracy: {}".format(float(success_count)/len(x_test)),end=' ')
            print("error: {}".format(float(error_count)/len(x_test)))
        return res


class Perceptron(Model):

    def __init__(self, *args, **kwargs):
        super(Perceptron, self).__init__(*args, **kwargs)
        self.model_name = 'Perceptron'

    def fit(self,x_train,y_train):
        w = np.random.uniform(-0.5, 0.5, (3, len(x_train[0])))
        eta = np.random.uniform(0.01, 0.4)
        for epoch in range(self.epochs):
            shuffle(x_train)
            shuffle(y_train)
            for x, y in zip(x_train, y_train):
                y_hat = np.argmax(np.dot(w, x))
                if y_hat != y:
                    w[int(y), :] = w[int(y), :] + eta * x
                    w[int(y_hat), :] = w[int(y_hat), :] - eta * x
        self.w = w
        return w


class PA(Model):
    def fit(self,x_train,y_train):
        # ToDo!!!!
        pass

class SVM(Model):

    def __init__(self, *args, **kwargs):
        super(SVM, self).__init__(*args, **kwargs)
        self.model_name = 'SVM'

    def fit(self,X,Y,folds=100):
        X = X.copy()
        Y = Y.copy()
        best_acc = -1
        alpha = np.random.random()
        eta = np.random.uniform(0.001, 0.5)

        for j in range(folds):

            if self.w is not None:
                w = self.w
            else:
                w = np.random.uniform(-0.5, 0.5, (3, len(X[0])))

            svm_w_list = []
            x_train, x_test, y_train, y_test = split_train_test_with_prop_random(X, Y)
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
            print(score)
            if score > best_acc:
                best_acc = score
                self.w = svm_w_list[idx]

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

def normalizeData(x_train):
    #x_train = stats.zscore(x_train)
    m, d = x_train.shape
    for i in range(d):
        cmean = x_train[:,i].mean()
        cstd = x_train[:,i].std()
        if cstd == 0:
            x_train[:,i] = 1.0 / float(m)
        else:
            x_train[:,i] = (x_train[:,i] - cmean) / cstd
    return x_train


def train_test_split(dataset, split=.7):
    train = list()
    train_size = split * len(dataset)
    dataset_copy = list(dataset)
    while len(train) < train_size:
        index = randrange(len(dataset_copy))
        train.append(dataset_copy.pop(index))
    return train, dataset_copy


def splitData(x_train, y_train):
    x_test = x_train[-int(np.round(0.3*len(x_train))):]
    x_train = x_train[:int(np.round(0.7*len(x_train)))]
    y_test = y_train[-int(np.round(0.3*len(y_train))):]
    y_train = y_train[:int(np.round(0.7*len(y_train)))]
    return x_train, x_test, y_train, y_test

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
    test_file_name = sys.argv[3]

    x_train, y_train = prepareData(train_x_file_name,train_y_file_name,test_file_name)
    x_train = min_max_scaling(x_train[:,:7])
    x_train, x_test, y_train, y_test = split_train_test_with_prop_random(x_train, y_train)

    model_names = ['Perceptron','SVM']
    predicts = {}
    for model_name in model_names:
        model = eval(model_name)(epochs=20)
        w = model.fit(x_train, y_train)
        predicts[model_name.lower()] = model.predict(x_test,y_test,to_print=True)

    # for i in range(len(x_test)):
    #     for model_name in model_names:
    #         if model_name.lower == 'svm':
    #             suffix = ''
    #         else:
    #             suffix = ', '
    #         print(model_name.lower() + ": " + str(predicts[model_name.lower()][i]) + suffix,end='')
    #     print()
     

    