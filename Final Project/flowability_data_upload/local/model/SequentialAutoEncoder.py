### Merge in
import numpy as np
import tensorflow as tf
import random

from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import ReLU
from tensorflow.keras.layers import BatchNormalization




def create_auto_encoder(encoded_width, sequence_length):
    input = layers.Input(shape=(29, 1000, 1))

    # Encoder
    x = layers.Conv2D(512, (3, 3), activation="relu", padding="same")(input)
    x = layers.MaxPooling2D((2, 2), padding="same")(x)
    x = layers.Conv2D(512, (3, 3), activation="relu", padding="same")(x)
    x = layers.MaxPooling2D((2, 2), padding="same")(x)

    # Decoder
    x = layers.Conv2DTranspose(512, (3, 3), strides=2, activation="relu", padding="same")(x)
    x = layers.Conv2DTranspose(512, (3, 3), strides=2, activation="relu", padding="same")(x)
    x = layers.Conv2D(1, (3, 3), activation="sigmoid", padding="same")(x)

    # Autoencoder
    autoencoder = Model(input, x)
    autoencoder.compile(optimizer="adam", loss="mse")
    autoencoder.summary()
    model = autoencoder
    return model


def create_auto_encoder_non_conv(encoded_width=17, sequence_length=1000, layer_width_multiplyer=256):
    # TODO: Non-conv does not work because input is 2D - Change Dense to Conv2D and MaxPooling Layers like above example
    n_features = 29
    input = layers.Input(shape=(n_features, sequence_length, 1))

    encode = Dense(n_features * 2 * layer_width_multiplyer)(input)
    encode = Dense(n_features * layer_width_multiplyer)(encode)
    encode = Dense(int(n_features / 2) * layer_width_multiplyer)(encode)
    encode = BatchNormalization()(encode)
    encode = ReLU()(encode)

    # Encoding Width
    encoded_features = Dense(encoded_width)(encode)

    decoder = Dense(int(n_features / 2) * layer_width_multiplyer)(encoded_features)
    decoder = Dense(n_features * layer_width_multiplyer)(decoder)
    decoder = Dense(n_features * 2 * layer_width_multiplyer)(decoder)

    output = Dense(input)

    model = Model(inputs=input, outputs=output)

    model.compile(optimizer='adam', loss='mse')
    return model


## train auto encoder
def train_auto_encoder(model, X, y, epochs=200, batch_size=64):
    model.fit(x=X,
              y=y,
              epochs=epochs,
              batch_size=batch_size,
              shuffle=True)
    return model


## Convert sequence of particles to specific size
def prep_sequence_length(X, sequence_length=2000):
    temp_X = X
    random.shuffle(temp_X)
    trimmed_X = X[sequence_length:]
    return trimmed_X


## Convert and save data to encoded format
def encode_data(model, X, y):
    encoded_data = model.predict(X)
    return encoded_data, y

## Calculate RMSE
def calculate_RMSE(model, X, y):
    # TODO: Implement RMSE for visualization
    RMSE = None
    return RMSE
