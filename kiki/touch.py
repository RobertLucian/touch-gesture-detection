import pandas as pd
import numpy as np
from tqdm import tqdm
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.metrics import categorical_accuracy

# read the dataset
file = '../datasets/generated_dataset.csv'
data = pd.read_csv(file)

# constants required for processing the dataset
timesteps = 11
features = 6
rows = data.shape[0]
batches = rows // timesteps
train_split_percentage = 0.8
traintest_split_point = int(batches * train_split_percentage)

# model parameters
batch_size = 8

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
model.add(LSTM(units=11, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(units=out_size, activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=[categorical_accuracy])

print('Train...')
model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=15,
          validation_data=(x_test, y_test))
score, acc = model.evaluate(x_test, y_test, batch_size=batch_size)

print('Test score:', score)
print('Test accuracy:', acc)