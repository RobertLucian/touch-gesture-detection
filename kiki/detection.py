import pandas as pd
import tensorflow as tf
import keras
import pkg_resources
from keras.models import Sequential, model_from_json
from keras.layers import Dense
from keras.layers import LSTM
from keras.metrics import categorical_accuracy

PACKAGE_NAME = 'kiki'

def get_resource_path(relative_path):
    """
    Get the absolute path to a given resource within the Python package.

    :param relative_path: A path to a resource (model, datasets).
    I.e "datasets/hitting.csv" or "models/touch.json".
    :return: The absolute path.
    """
    FILENAME_PATH = pkg_resources.resource_filename(PACKAGE_NAME, relative_path)
    return FILENAME_PATH

def save_trained_model(model_path, model):
    """
    Save a trained model to disk.

    :param model_path: A relative/absolute path to the output model. Must not
    include and extension because there are 2 files written to disk: a json and an h5.
    I.e: "output/generic_model".
    :param model: The actual trained keras.models.Model object.
    :return: Nothing.
    """
    json = model.to_json(indent=2)
    with open(model_path + '.json', 'w') as json_file:
        json_file.write(json)
    model.save_weights(model_path + '.h5')

def load_pretrained_model(model_path, inpackage_data=False):
    """
    Load a pre-trained model.

    :param model_path: Relative path to an in-built resource of the package
    or an absolute path. Both the model and the weights must have the same name
    i.e "generic_model.json" and "generic_model.h5". When passing the argument,
    don't add the extension of it, because we're referring to two related files.
    :param inpackage_data: If the data is located within the package or not.
    :return: A keras.models.Model model.
    """
    if inpackage_data:
        model_path = get_resource_path(model_path)

    with open(model_path + '.json', 'r') as f:
        loaded_json = f.read()
    loaded_model = model_from_json(loaded_json)
    loaded_model.load_weights(model_path + '.h5')

    return loaded_model


def train_touchsensor_model(
        dataset='datasets/generated_dataset.csv',
        inpackage_data=True,
        timesteps=11,
        features=6,
        batch_size=64,
        lstm_units=32,
        epochs=30,
        cuda_gpu=False
):
    """
    Train the neural network against the dataset containing gesture/motion recordings.
    The model of the neural network is already designed to be used with this kind of data.

    :param dataset: Relative path to an in-package resource or an absolute path.
    :param inpackage_data: Whether the data is located within the package.
    :param timesteps: How many timesteps there are in a single sample.
    :param features: Number of features taken into consideration for each sample.
    :param batch_size: After how many predictions the weights get updated.
    :param lstm_units: Number of how many LSTM memory units there are.
    :param epochs: Number of epochs the neural network gets trained for.
    :param cuda_gpu: Whether to use an Nvidia GPU or not. Must have backend support for GPU,
    like use tensorflow-gpu for instance.
    :return: Model, score and accuracy.
    """
    # use gpu if enabled and available
    if cuda_gpu:
        config = tf.ConfigProto( device_count = {'GPU': 1 , 'CPU': 6} )
        sess = tf.Session(config=config)
        keras.backend.set_session(sess)


    # read the dataset
    print('Read touchsensor dataset')
    if inpackage_data:
        file = get_resource_path(dataset)
    else:
        file = dataset
    data = pd.read_csv(file)


    # constants required for processing the dataset
    print('Preprocess the dataset')
    rows = data.shape[0]
    batches = rows // timesteps
    train_split_percentage = 0.8
    traintest_split_point = int(batches * train_split_percentage)

    # do the one-hot-encoding for the output
    data['Label'] = pd.Categorical(data['Label'])
    data_dummies = pd.get_dummies(data['Label'])
    data = pd.concat([data, data_dummies], axis=1)
    data.drop('Label', inplace=True, axis=1)

    # split the dataset into 80/20 splits
    train_data = data.iloc[0:traintest_split_point * timesteps]
    test_data = data.iloc[traintest_split_point * timesteps:]

    # separate labels from data for the training set
    x_train = train_data.loc[:, 'A':'F'].to_numpy().reshape((-1, timesteps, features)) / 4095
    y_train = train_data.drop(list('ABCDEF'), axis=1)[(train_data.index + 1) % timesteps == 0].to_numpy()

    # separate labels from data for the test set
    x_test = test_data.loc[:, 'A':'F'].to_numpy().reshape((-1, timesteps, features)) / 4095
    y_test = test_data.drop(list('ABCDEF'), axis=1)[(test_data.index + 1) % timesteps == 0].to_numpy()

    # no output labels
    out_size = y_test.shape[1]

    print('Build model...')
    model = Sequential()
    model.add(LSTM(units=lstm_units, dropout=0.1, recurrent_dropout=0.1))
    model.add(Dense(units=out_size, activation='softmax'))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=[categorical_accuracy])

    print('Train...')
    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              validation_data=(x_test, y_test))
    score, acc = model.evaluate(x_test, y_test, batch_size=batch_size)

    print('Test score:', score)
    print('Test accuracy:', acc)

    return model, score, acc