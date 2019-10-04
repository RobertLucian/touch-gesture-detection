import pandas as pd
import tensorflow as tf
import keras
import pkg_resources
from keras.models import Sequential, model_from_json
from keras.layers import Dense
from keras.layers import LSTM
from keras.metrics import categorical_accuracy

PACKAGE_NAME = 'kik'

def get_resource_path(relative_path):
    FILENAME_PATH = pkg_resources.resource_filename(PACKAGE_NAME, relative_path)
    return FILENAME_PATH

def load_pretrained_model(model_path, inpackage_data=False):
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