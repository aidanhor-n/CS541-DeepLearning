from keras.models import Sequential
from keras.layers import Dense


def create_model(input_features=16, network_width=16, hidden_layers=1, activation='relu'):
    """
    :param input_features: number of encoded features
    :param network_width: the width of the network per hidden layer
    :param hidden_layers: the number of extra hidden layers to
        include (there are atleast 2 hidden layers if hidden_layers=0)

    :param activation: activation function to use within keras for the hidden layers

    :return: a compiled but untrained keras model

    create flowability classification model

    """

    model = Sequential()
    # Larger first layer
    model.add(Dense(2 * network_width, input_dim=input_features, activation=activation))
    # add hidden layers
    for hidden_layer in range(hidden_layers):
        model.add(Dense(network_width, activation=activation))
    # add half width final hidden layer
    model.add(Dense(0.5 * network_width, activation=activation))
    # Sigmoid because binary classification
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


def train_model(model, X, y, epochs=256, batch_size=128):
    model.fit(X, y, epochs=epochs, batch_size=batch_size, verbose=0)
    return model


def make_predictions(model, X):
    predictions = model.predict_classes(X)
    return predictions


def get_accuracy(x_test, y_test, model):
    accuracy = 0
    return accuracy
