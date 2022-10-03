import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import scipy

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
    Y=Y.astype(int)
    for i in range(y.shape[0]):
        c.append(np.log(yhat[i, Y[i,0]]))
    ce=-np.mean(c)
    # print(ce)
    return ce

def relu(z):
    z[z<0]=0
    return z


def forward_prop (x, y, weightsAndBiases, iteration = 0, printFlag = True):
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
    if printFlag:
        print(f'{iteration:>5}{"|":*<{int(np.exp(loss))}}')
    # Return loss, pre-activations, post-activations, and predictions
    return loss, zs, hs, yhat

# relu' function
def d_relu(z):
    z[z>0]=1
    z[z<=0]=0
    return z
def back_prop (x, y, weightsAndBiases, iteration = 0):
    loss, zs, hs, yhat = forward_prop(x, y, weightsAndBiases, iteration)
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
    loss, zs, hs, yhat=forward_prop(X,Y,weightsAndBiases, "FINAL")
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
        print(f"\n=== EPOCH {epoch} ===\n")
        print(f"iter | exp(loss)")
        for j in range(iter):
            #print(f"Iteration {j}:")
            # print(j)
            # print(mini_batch)
        # TODO: implement SGD.
            Xtr=trainX[j*mini_batch:(j+1)*mini_batch,:]
            ytr=trainY[j*mini_batch:(j+1)*mini_batch,:]
            dWdb,dJdWs,dJdbs = back_prop(Xtr, ytr, weightsAndBiases, j)
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

    def toyFunction (x1, x2):
        xy=[x1,x2]
        xy_inverse=pca.inverse_transform(xy)
        loss= forward_prop (trainX, trainY, xy_inverse, f"({x1 : .1f},{x2: .1f})")[0]
        return loss

    #fig = plt.figure()
    ax = plt.axes(projection='3d')

    # scaler= StandardScaler()
    # scaler.fit(trajectory)
    # ws=scaler.transform(trajectory)

    # Performing PCA
    pca=PCA(n_components=2)
    xy=pca.fit_transform(trajectory)

    #Creating the mesh grid for surface plot
    x=xy[:,0]
    y=xy[:,1]
    axis1 = np.linspace(np.min(x),np.max(x), 20)
    axis2 = np.linspace(np.min(y),np.max(y), 20)
    Xaxis, Yaxis = np.meshgrid(axis1, axis2)
    Zaxis = np.zeros((len(axis1), len(axis2)))

    print("\n=== Generating Loss Curve ===\n")
    for i in range(len(axis1)):
        for j in range(len(axis2)):
            #Calcualting the Loss Value
            Zaxis[i,j] = toyFunction(Xaxis[i,j], Yaxis[i,j])

    ax.plot_surface(Xaxis, Yaxis, Zaxis, alpha=0.6)

    # Now superimpose a scatter plot showing the weights during SGD.
    X=[]
    Y=[]
    Z=[]
    print("\n=== Plotting SGD Trajectory ===\n")
    for i in range(0,len(x)-1):
        #Creating 1 point between each epoch
        num_points = 1
        tempx=np.linspace(x[i],x[i+1],num_points)
        tempy=np.linspace(y[i],y[i+1],num_points)
        for j in range(num_points):
            X.append(tempx[j])
            Y.append(tempy[j])
            #Calculating the Loss value
            Z.append(toyFunction(tempx[j], tempy[j])) 
    
    ax.scatter(X,Y,Z, color='r')
    plt.show()

def split_data():
    X = np.reshape(np.load("Homework 4/fashion_mnist_train_images.npy"), (-1, 28 * 28)) / 255
    y = np.reshape(np.load("Homework 4/fashion_mnist_train_labels.npy"), (-1, 1))
    data = np.concatenate((X, y), axis=1)
    np.random.shuffle(data)
    n = int(0.20 * X.shape[0])
    data_tr, data_vali = data[n:, :], data[:n, :]
    return data_tr, data_vali

class hyperparameter:
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
    mini_batchs=[64,256]
    learning_rates=[0.001,0.01]
    epochs=[10,20]
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
                                best_parameter=hyperparameter(num_hidden_layer,num_hidden,lr,mini_batch,epoch,alpha)
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

    xtest=np.reshape(np.load("Homework 4/fashion_mnist_train_images.npy"), (-1, 28 * 28)) / 255
    ytest = np.reshape(np.load("Homework 4/fashion_mnist_train_labels.npy"), (-1, 1))
    ytest = enc.fit_transform(ytest)

    # # Initialize weights and biases randomly
    weightsAndBiases = initWeightsAndBiases()

    # find best hypterparameters
    # hP=findBestHyperparameters(trainX,trainY,validX,validY)
    # print(f"Best Hypterparameter tuning:\nnum_hidden_layer: {hP.num_hidden_layer}\nnum_hidden: {hP.num_hidden}\nlr: {hP.lr}\nmini_batch: {hP.mini_batch}\nepoch: {hP.epoch}\nalpha: {hP.alpha}")

    # Perform gradient check on random training examples
    #print(scipy.optimize.check_grad(lambda wab: forward_prop(np.atleast_2d(trainX[0:5,:]), np.atleast_2d(trainY[0:5,:]), wab)[0], lambda wab: back_prop(np.atleast_2d(trainX[0:5,:]), np.atleast_2d(trainY[0:5,:]), wab), weightsAndBiases))

    # Training and Evaluation
    weightsAndBiases,trajectory=train(trainX, trainY, weightsAndBiases,mini_batch=16,lr=0.001,alpha=0.01,epoch=10)
    acc,ce=evaluation(xtest,ytest,weightsAndBiases)
    #print(acc)

    # Plot the SGD trajectory
    #plotSGDPath(trainX, trainY, trajectory)
