import numpy as np
import itertools
from sklearn.model_selection import train_test_split
from cmath import inf
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder

# Grid search
grid = {
    'batch_size': [50, 100, 150, 200],
    'learning_rate':  [1E-5, 1E-4, 1E-3, 1E-2],
    'epochs': [5, 10, 15, 20],
    'regularization':  [1E-5, 1E-4, 1E-3, 1E-2]
}


# load datasets
X_tr = np.reshape(np.load("fashion_mnist_train_images.npy"), (-1, 28 * 28)) / 255
ytr = np.reshape(np.load("fashion_mnist_train_labels.npy"), (-1, 1))
X_te = np.reshape(np.load('fashion_mnist_test_images.npy'), (-1, 28 * 28)) / 255
y_te = np.reshape(np.load('fashion_mnist_test_labels.npy'), (-1, 1))

# set aside 20% of training data to be the validation set
X_train, X_val, y_train, y_val = train_test_split(X_tr, ytr, test_size=0.2)

# optimized hyperparameters
opt_epoch = 0
opt_batch_size = 0
opt_learning_rate = 0
opt_regularization = 0

def onehotencoding(y):
    enc_y=OneHotEncoder().fit_transform(y)
    return enc_y

def z(X,w,b):
    # z: pre-activation scores
    z=X.dot(w) + b

    # normalization
    normalized = np.exp(z)/np.sum(np.exp(z),axis=1).reshape(-1,1)
    return normalized


def cross_entropy(X,y,w,b,alpha):
    w = np.asarray(w)
    y_hat = np.exp(z(X,w,b)) / sum(np.exp(z(X,w,b)))

    c = []
    for i in range(X.shape[0]):
        c.append(np.log(y_hat[i,int(y[i])]))

    ce =- np.mean(c)+alpha/2*np.mean(w.dot(w.T))
    return ce


def train(x_train, y_train, learning_rate, alpha, batch_size, num_epochs):
    y_train = onehotencoding(y_train)
    w = np.random.rand(x_train.shape[1], y_train.shape[1])
    b = np.random.rand(1, y_train.shape[1])    
    
    for i in range(num_epochs):
        batch_index = 0

        for j in range(int(x_train.shape[0]/batch_size - 1)):
            start_index = batch_index * batch_size
            end_index = (batch_index+1) * (batch_size) - 1

            X_batch = x_train[start_index:end_index]
            y_batch = y_train[start_index:end_index]

            batch_index += 1
                
            grad_w=X_batch.T.dot(z(X_batch,w,b)-y_batch)
            w = (1-alpha*learning_rate) * w - learning_rate * grad_w
            if np.abs(learning_rate * (z(X_batch,w,b) - y_batch).mean()) < np.inf:
                b -= learning_rate * (z(X_batch,w,b) - y_batch).mean()
    return w,b


def softmax_sgd():
    optimized_weights = []
    optimized_bias = 0
    min_cost=-1

    # use grid search
    for values in itertools.product(*grid.values()):
        point = dict(zip(grid.keys(), values))
        epoch = point['epochs']
        batch_size = point['batch_size']
        learning_rate = point['learning_rate']
        alpha = point['regularization']

        weights, bias = train(X_train, y_train, learning_rate, alpha, batch_size, epoch)
        ce = cross_entropy(X_val, y_val, weights, bias, alpha)

        if min_cost==-1 or ce < min_cost:
            min_cost = ce
            opt_batch_size = batch_size
            opt_learning_rate = learning_rate
            opt_epoch = epoch
            opt_regularization = alpha
            optimized_weights = weights
            optimized_bias = bias
    return min_cost, opt_epoch, opt_batch_size, opt_learning_rate, opt_regularization, optimized_weights, optimized_bias



min_cost, epoch, batch_size, learning_rate, regularization, optimized_weights, optimized_bias=softmax_sgd()
print("Optimized hyperparameters:")
print("     Mini-batch size: " + str(batch_size))
print("     Learning rate: " + str(learning_rate))
print("     # of epochs: " + str(epoch))
print("     Alpha: " + str(regularization))
print("     Weights: " + str(optimized_weights))
print("         " + str(optimized_weights))
print("     Bias: ")
print("         " + str(optimized_bias))
print("     min_cost on validation data: " + str(min_cost))
print("\n")


print("---------------TESTING-------------------")
# ------Use optimized hyperparams to fit test data------
# Choose the largest bit as predicting class
def argmax(yhat):
    return np.argmax(np.asarray(yhat),axis=1)

# y_hat for test data
y_hat=argmax(z(X_te,optimized_weights,optimized_bias)).reshape(-1,1)

# Cross entropy for test data
ce = cross_entropy(X_te, y_te, optimized_weights, optimized_bias, regularization)
print("Cross Enropy: " + str(ce))

# Percent correctly classified example
percent = accuracy_score(y_te, y_hat) * 100

print("% of correctly classified images: " + str(percent) + "%")
