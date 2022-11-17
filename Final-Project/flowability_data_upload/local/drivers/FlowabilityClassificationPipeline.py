import os

from flowability_data_upload.local.generate_datasets.VirtualPowderMixing import generate_virtually_mixed_powders
from flowability_data_upload.local.model.SequentialAutoEncoder import create_auto_encoder, train_auto_encoder, prep_sequence_length, encode_data, calculate_RMSE
from flowability_data_upload.local.model.PowderFlowabilityClassification import create_model, train_model, make_predictions, get_accuracy


def check_gen_virtual_data():
    if os.path.isdir("./virtual_data"):
        path, dirs, files = next(os.walk("./virtual_data"))
        if len(files) != 0:
            return True
    else:
        return True
    os.mkdir("./virtual_data")
    generate_virtually_mixed_powders()


def auto_encoder(X, y, x_test, y_test, encoded_features=17,sequence_length=1000):
    model = create_auto_encoder(encoded_features, sequence_length)
    model = train_auto_encoder(model, X)
    X_trimmed = prep_sequence_length(X, sequence_length)
    X_test_trimmed = prep_sequence_length(x_test, sequence_length)
    X_encoded, y = encode_data(model, X_trimmed, y)
    X_test_encoded, y_test_encoded = encode_data(model, X_test_trimmed, y_test)
    return X_encoded, y, X_test_encoded, y_test_encoded, model


def flowability_classification(X, y, input_features=16, network_width=16, hidden_layers=1, activation='relu'):
    model = create_model(input_features, network_width, hidden_layers, activation)
    model = train_model(model, X, y, epochs=256, batch_size=128)
    return model


def experiment_encoded_features_auto_encoders():
    x_train, x_test, y_train, y_test = load_virtual_data()
    rmse_list = list()
    for encoded_features in range(1,25):
        x_train, x_test, y_train, y_test, model = auto_encoder(x_train, y_train)
        rmse = calculate_RMSE(model, x_test, y_test)
        rmse_list.append([encoded_features, rmse])
    return rmse_list


def experiment_encoded_features_flowability_classification():
    x_train, x_test, y_train, y_test = load_virtual_data()
    accuracy_list = list()
    for encoded_features in range(1, 25):
        x_train, x_test, y_train, y_test, model = auto_encoder(x_train, y_train)
        flowability_classification(x_train, y_train)
        accuracy = get_accuracy(model, x_test, y_test)
        accuracy_list.append([encoded_features, accuracy])
    return accuracy_list

# !!! What does this do??
def load_virtual_data():
    x_train = x_test = y_train = y_test = None
    return x_train, x_test, y_train, y_test