import numpy as np
from sklearn.utils import shuffle
from scipy import stats
import sys


def prepareData(train_x_file_name,train_y_file_name,test_file_name):
    arr = np.loadtxt("train_x.txt", usecols=(1,2,3,4,5,6,7), dtype=np.float32, delimiter=',')
    df = list()
    with open("train_x.txt",'r') as file:
        for i,line in enumerate(file):
            if line[0] == 'M':
                appendcol = [1.,0.,0.]        
            elif line[0] == 'F':
                appendcol = [0.,1.,0.]        
            elif line[0] == 'I':
                appendcol = [0.,0.,1.]
            df.append(np.append(arr[i],appendcol))
    x_train = np.array(df).astype('float32')
    y_train = np.loadtxt("train_y.txt")
    return x_train, y_train, x_test

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


def splitData(x_train, y_train):
    
    x_test = x_train[-int(np.round(0.3*len(x_train))):]
    x_train = x_train[:int(np.round(0.7*len(x_train)))]
    y_test = y_train[-int(np.round(0.3*len(y_train))):]
    y_train = y_train[:int(np.round(0.7*len(y_train)))]

    return x_train, x_test, y_train, y_test


def perceptron_fit(x_train, y_train):
    w = np.random.uniform(-0.5,0.5,(3,len(x_train[0])))
    eta = np.random.uniform(0.01,0.4)
    for epoch in range(10):
        x_train, y_train = shuffle(x_train,y_train,random_state = 1) 
        for x,y in zip(x_train,y_train):
            y_hat = np.argmax(np.dot(w,x))
            if y_hat != y:
                w[int(y),:] = w[int(y),:] + eta*x
                w[int(y_hat),:] = w[int(y_hat),:] - eta*x
    return w
    

def svm_fit(x_train, y_train):
    w = np.random.uniform(-0.5,0.5,(3,len(x_train[0])))
    alpha = np.random.random()
    eta = np.random.uniform(0.001,0.5)
    svm_w_list = []
    for epoch in range(10):
        x_train, y_train = shuffle(x_train,y_train,random_state = 1) 
        for x,y in zip(x_train,y_train):
            y_hat = np.argmax(np.dot(w,x))
            if y_hat != y:
                w[int(y),:] = (1-eta*alpha)*w[int(y),:] + eta*x
                w[int(y_hat),:] = (1-eta*alpha)*w[int(y_hat),:] - eta*x
            for i in range(w.shape[0]):
                if i != y and i !=y_hat:
                    w[i,:] = (1-eta*alpha)*w[i,:]
        svm_w_list.append(w)
    retrun svm_w_list


def validate(w_list, x_valid, y_valid):

    w_dict = {}
    for w in w_list:
        w_dict[w] = 0
        for t in range(0,len(x_valid)):
            y_hat = np.argmax(np.dot(w,x_valid[t]))
            if y_hat == y_valid[t]:
                w_dict[w] = w_dict[w] + 1
    return max(w_dict.items(), key=operator.itemgetter(1))[0]

def svm_predict(w_svm, x_test, y_test):

    e_perceptron = 0
    c_perceptron = 0
    for t in range(0,len(x_test)):
        y_hat = np.argmax(np.dot(w_svm,x_test[t]))
        if y_hat != y_test[t]:
            e_perceptron = e_perceptron + 1
        else:
            c_perceptron = c_perceptron + 1
    print("svm error: {}".format(float(e_perceptron)/len(x_test)))
    print("svm correct: {}".format(float(c_perceptron)/len(x_test)))
 
	
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
    x_train, x_test, y_train, y_test = splitData(x_train, y_train)
    w = svm_fit(x_train, y_train)
    svm_predict(w, x_test, y_test)
     

    