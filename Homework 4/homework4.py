import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import scipy.optimize

# For this assignment, assume that every hidden layer has the same number of neurons.
NUM_HIDDEN_LAYERS = 3
NUM_INPUT = 784
NUM_HIDDEN = 40
NUM_OUTPUT = 10
from sklearn.preprocessing import OneHotEncoder
global enc
enc = OneHotEncoder(sparse=False)


# Performs a standard form of random initialization of weights and biases
def initWeightsAndBiases ():
    Ws = []
    bs = []

    np.random.seed(0)
    W = 2*(np.random.random(size=(NUM_INPUT, NUM_HIDDEN))/NUM_INPUT**0.5) - 1./NUM_INPUT**0.5
    Ws.append(W)
    b = 0.01 * np.ones(NUM_HIDDEN)
    bs.append(b)

    for i in range(NUM_HIDDEN_LAYERS - 1):
        W = 2*(np.random.random(size=(NUM_HIDDEN, NUM_HIDDEN))/NUM_HIDDEN**0.5) - 1./NUM_HIDDEN**0.5
        Ws.append(W)
        b = 0.01 * np.ones(NUM_HIDDEN)
        bs.append(b)

    W = 2*(np.random.random(size=(NUM_HIDDEN, NUM_OUTPUT))/NUM_HIDDEN**0.5) - 1./NUM_HIDDEN**0.5
    Ws.append(W)
    b = 0.01 * np.ones(NUM_OUTPUT)
    bs.append(b)
    return np.hstack([ W.flatten() for W in Ws ] + [ b.flatten() for b in bs ])

# Unpack a list of weights and biases into their individual np.arrays.
def unpack (weightsAndBiases):
    # Unpack arguments
    Ws = []
    # Weight matrices
    start = 0
    end = NUM_INPUT*NUM_HIDDEN
    W = weightsAndBiases[start:end]
    Ws.append(W)

    for i in range(NUM_HIDDEN_LAYERS - 1):
        start = end
        end = end + NUM_HIDDEN*NUM_HIDDEN
        W = weightsAndBiases[start:end]
        Ws.append(W)

    start = end
    end = end + NUM_HIDDEN*NUM_OUTPUT
    W = weightsAndBiases[start:end]
    Ws.append(W)

    Ws[0] = Ws[0].reshape(NUM_INPUT, NUM_HIDDEN)
    for i in range(1, NUM_HIDDEN_LAYERS):
        # Convert from vectors into matrices
        Ws[i] = Ws[i].reshape(NUM_HIDDEN, NUM_HIDDEN)
    Ws[-1] = Ws[-1].reshape(NUM_HIDDEN, NUM_OUTPUT)

    # Bias terms
    bs = []
    start = end
    end = end + NUM_HIDDEN
    b = weightsAndBiases[start:end]
    bs.append(b)

    for i in range(NUM_HIDDEN_LAYERS - 1):
        start = end
        end = end + NUM_HIDDEN
        b = weightsAndBiases[start:end]
        bs.append(b)

    start = end
    end = end + NUM_OUTPUT
    b = weightsAndBiases[start:end]
    bs.append(b)

    return Ws, bs

from scipy.special import softmax
# compute wx+b
def computez(X,w,b):
    z=X.dot(w) + b
    return z
# Softmax layer to choose the larget bit as predicting class
def argmax(yhat):
    return np.argmax(np.asarray(yhat),axis=1)

def softmax(z):
    return np.exp(z)/np.sum(np.exp(z),axis=1).reshape(-1,1)

def cost(yhat,y):
    Y=enc.inverse_transform(y)
    c=[]
    Y=Y.astype(np.int)
    for i in range(y.shape[0]):
        c.append(np.log(yhat[i, Y[i,0]]))
    ce=-np.mean(c)
    # print(ce)
    return ce

def relu(z):
    z[z<0]=0
    return z


def forward_prop (x, y, weightsAndBiases):
    Ws, bs = unpack(weightsAndBiases)
    zs, hs = [], []
    h = x
    for i in range(len(Ws)-1):
        zhat = computez(h, np.asarray(Ws[i]), np.asarray(bs[i]).reshape(1,-1))
        h=relu(zhat)
        zs.append(zhat)
        hs.append(h)
    zhat = computez(h, np.asarray(Ws[-1]), np.asarray(bs[-1]).reshape(1,-1))
    zs.append(zhat)
    yhat = softmax(zhat)
    # print(yhat)
    loss = cost(yhat, y)
    print("Training Loss:"+str(loss))
    # Return loss, pre-activations, post-activations, and predictions
    return loss, zs, hs, yhat

# relu' function
def d_relu(z):
    z[z>0]=1
    z[z<=0]=0
    return z
def back_prop (x, y, weightsAndBiases):
    loss, zs, hs, yhat = forward_prop(x, y, weightsAndBiases)
    Ws,bs=unpack(weightsAndBiases)

    dJdWs = []  # Gradients w.r.t. weights
    dJdbs = []  # Gradients w.r.t. biases
    dfdz = (yhat - y)
    # TODO
    for i in range(NUM_HIDDEN_LAYERS, 0, -1):
        pass
        # TODO
        dfdw = hs[i-1].T.dot(dfdz)
        # print(dfdw.shape)
        dfdb = dfdz
        dJdWs.append(dfdw)
        dJdbs.append(dfdb)
        # compute dfdz for previous layer
        dfdh = dfdz.dot((Ws[i].T))
        # print(zs[i-1])
        # print(d_relu(zs[i-1]))
        dfdz = np.multiply(dfdh, d_relu(zs[i-1]))
    # first layer
    dfdw = x.T.dot(dfdz)
    dfdb = dfdz
    dJdWs.append(dfdw)
    dJdbs.append(dfdb)

    # reverse the DJDW DJDB list
    dJdWs = dJdWs[::-1]
    dJdbs = dJdbs[::-1]
    # for i in dJdbs:
    #     print(i.shape)
    # Concatenate gradients
    return np.hstack([ dJdW.flatten() for dJdW in dJdWs ] + [ dJdb.flatten() for dJdb in dJdbs ]),dJdWs,dJdbs


def update_weights(weightsAndBiases,dJdWs,dJdbs,alpha,lr):
    Ws, bs = unpack(weightsAndBiases)
    layers=len(Ws)
    # print(layers)
    for i in range(layers):
        Ws[i] = (1 - alpha * lr) * Ws[i] - lr * dJdWs[i]
        bs[i]-=lr*dJdbs[i].mean()
    weightsAndBiases=np.hstack([W.flatten() for W in Ws] + [b.flatten() for b in bs])
    # forward_prop(x, y, weightsAndBiases)
    return weightsAndBiases

def evaluation(X,Y,weightsAndBiases):
    loss, zs, hs, yhat=forward_prop(X,Y,weightsAndBiases)
    y_pre=argmax(yhat)
    ce=cost(yhat,Y)
    Y=enc.inverse_transform(Y)
    from sklearn.metrics import accuracy_score
    acc=accuracy_score(Y,y_pre)
    print("Accuracy:"+str(acc))
    return acc,ce

def train(trainX, trainY, weightsAndBiases,lr,mini_batch,alpha,epoch):
    NUM_EPOCHS = epoch
    trajectory = []
    iter = int(trainX.shape[0] / mini_batch)
    for epoch in range(NUM_EPOCHS):
        for j in range(iter):
            print("Iteration "+str(j)+":")
            # print(j)
            # print(mini_batch)
        # TODO: implement SGD.
            Xtr=trainX[j*mini_batch:(j+1)*mini_batch,:]
            ytr=trainY[j*mini_batch:(j+1)*mini_batch,:]
            dWdb,dJdWs,dJdbs = back_prop(Xtr, ytr, weightsAndBiases)
            weightsAndBiases=update_weights(weightsAndBiases,dJdWs,dJdbs,alpha,lr)
            trajectory.append(weightsAndBiases)
        # TODO: save the current set of weights and biases into trajectory; this is
        # useful for visualizing the SGD trajectory.
    return weightsAndBiases, trajectory



def plotSGDPath (trainX, trainY, trajectory):
    # TODO: change this toy plot to show a 2-d projection of the weight space
    # along with the associated loss (cross-entropy), plus a superimposed
    # trajectory across the landscape that was traversed using SGD. Use
    # sklearn.decomposition.PCA's fit_transform and inverse_transform methods.
    ce,wb=[],[]
    #surfaceplot
    # preparing the data
    print(len(trajectory))
    for weightsAndBiases in trajectory:
        loss, _, _, _ = forward_prop(trainX, trainY, weightsAndBiases)
        ce.append(loss)
        wb.append(weightsAndBiases)
    wb=np.asarray(wb)
    ce=np.asarray(ce).reshape(-1,1)

    # PCA
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2)
    wb=pca.fit_transform(wb)
    # print(wb.shape)

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    Xaxis, Yaxis = np.meshgrid(wb[:,0], wb[:,1])
    ax.plot_surface(Xaxis, Yaxis, ce, alpha=0.6)  # Keep alpha < 1 so we can see the scatter plot too.

    # scatter plot
    from random import sample
    trajectory=sample(trajectory,2500)
    print(len(trajectory))
    ce,wb=[],[]
    for weightsAndBiases in trajectory:
        loss, _, _, _ = forward_prop(trainX, trainY, weightsAndBiases)
        ce.append(loss)
        wb.append(weightsAndBiases)
    wb = np.asarray(wb)
    ce = np.asarray(ce).reshape(-1, 1)

    # PCA
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2)
    wb = pca.fit_transform(wb)
    # print(wb.shape)
    ax.scatter(wb[:,0], wb[:,1], ce, color='r')


    plt.show()

def split_data():
    X = np.reshape(np.load("fashion_mnist_train_images.npy"), (-1, 28 * 28)) / 255
    y = np.reshape(np.load("fashion_mnist_train_labels.npy"), (-1, 1))
    data = np.concatenate((X, y), axis=1)
    np.random.shuffle(data)
    n = int(0.20 * X.shape[0])
    data_tr, data_vali = data[n:, :], data[:n, :]
    return data_tr, data_vali

class parameter:
    def __init__(self,num_hidden_layer,num_hidden,lr,mini_batch,epoch,alpha):
        self.num_hidden_layer=num_hidden_layer
        self.num_hidden=num_hidden
        self.lr=lr
        self.mini_batch=mini_batch
        self.epoch=epoch
        self.alpha=alpha

def findBestHyperparameters(trainX, trainY,validX,validY):
    num_hidden_layers=[3,4,5]
    num_hiddens=[30,40]
    mini_batchs=[20,30]
    learning_rates=[0.001,0.01]
    epochs=[20,50]
    alphas=[0.001,0.01]
    # num_hidden_layers=[3,4,5]
    # num_hiddens=[40]
    # mini_batchs=[20]
    # learning_rates=[0.001]
    # epochs=[20]
    # alphas=[0.01]


    min_cost=-1
    best_parameter=[]
    for num_hidden_layer in num_hidden_layers:
        for num_hidden in num_hiddens:
            for mini_batch in mini_batchs:
                for lr in learning_rates:
                    for epoch in epochs:
                        for alpha in alphas:
                            # change the NN settings
                            global NUM_HIDDEN_LAYERS,NUM_HIDDEN
                            NUM_HIDDEN_LAYERS = num_hidden_layer
                            NUM_HIDDEN = num_hidden
                            # initialize the weights and bias
                            weightsAndBiases = initWeightsAndBiases()
                            weightsAndBiases,_=train(trainX, trainY, weightsAndBiases,lr,mini_batch,alpha,epoch)
                            acc,ce=evaluation(validX,validY,weightsAndBiases)
                            if min_cost == -1 or ce < min_cost:
                                min_cost = ce
                                best_parameter=parameter(num_hidden_layer,num_hidden,lr,mini_batch,epoch,alpha)
                                # best_wb=weightsAndBiases
    return best_parameter
if __name__ == "__main__":
    # TODO: Load data and split into train, validation, test sets
    data_tr, data_vali = split_data()
    trainX = data_tr[:, :-1]
    trainY = data_tr[:, -1].reshape(-1, 1)
    trainY=enc.fit_transform(trainY)

    validX = data_vali[:, :-1]
    validY = data_vali[:, -1].reshape(-1, 1)
    validY = enc.fit_transform(validY)

    xtest=np.reshape(np.load("fashion_mnist_train_images.npy"), (-1, 28 * 28)) / 255
    ytest = np.reshape(np.load("fashion_mnist_train_labels.npy"), (-1, 1))
    ytest = enc.fit_transform(ytest)

    # # Initialize weights and biases randomly
    weightsAndBiases = initWeightsAndBiases()

    # find best hypterparameters
    # hyperpara=findBestHyperparameters(trainX,trainY,validX,validY)
    # print("Best Hypterparameters setting:(num_hidden_layer,num_hidden,lr,mini_batch,epoch,alpha)")
    # print(hyperpara.num_hidden_layer,hyperpara.num_hidden,hyperpara.lr,hyperpara.mini_batch,hyperpara.epoch,hyperpara.alpha)

    # # Perform gradient check on random training examples
    # print(scipy.optimize.check_grad(lambda wab: forward_prop(np.atleast_2d(trainX[0:5,:]), np.atleast_2d(trainY[0:5,:]), wab)[0], \
    #                                 lambda wab: back_prop(np.atleast_2d(trainX[0:5,:]), np.atleast_2d(trainY[0:5,:]), wab), \
    #                                 weightsAndBiases))

    # Training and Evaluation
    weightsAndBiases,trajectory=train(trainX, trainY, weightsAndBiases,mini_batch=20,lr=0.0005,alpha=0.01,epoch=20)
    acc,ce=evaluation(xtest,ytest,weightsAndBiases)
    print(acc)

    # Plot the SGD trajectory
    plotSGDPath(trainX, trainY, trajectory)
