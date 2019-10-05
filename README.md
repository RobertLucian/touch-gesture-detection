# Touch Gesture Detection [![Build Status](https://travis-ci.org/RobertLucian/touch-gesture-detection.svg?branch=master)](https://travis-ci.org/RobertLucian/touch-gesture-detection) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://robertlucian.mit-license.org)
A concept library for detecting gestures from an array of touch sensors (and possibly other sensors as well).

## Installing

To install the library, please be sure you've got Python 3.6+ and run
```bash
pip install git+https://github.com/RobertLucian/touch-gesture-detection.git
```

## Development

To develop it, clone the repo and run 
```bash
virtualenv -p python3 env # make sure you've got python 3.6+ installed

env\Scripts\activate # for Windows-based systems
source env/bin/activate # for Linux-based systems

pip install --editable .
```

## API

`kiki.detection.get_resource_path` - function to get the absolute file path
of a file that resides in the `kiki` package. Must use a relative path for it.

`kiki.detection.load_pretrained_model` - function to load a pre-trained model.
Can specify whether the given path to the dataset is for a in-packaged model 
or just another that's coming out of it.

`kiki.detection.train_touchsensor_model` - function to train the touch-gesture
detection mechanism. Takes in a dataset and sets some important hyper-parameters.
This function should be deprecated at some point and instead have the process of training new
models streamlined. To train new motions, just add more data to the dataset.

To see the parameters of these functions, check the embedded docstrings of each one.

## Dataset

Within the package, there are a couple of datasets included:
* `datasets/vertical_swipe.csv`
* `datasets/horizontal_swipe.csv`
* `datasets/tapping.csv`
* `datasets/double_tapping.csv`
* `datasets/hitting.csv` - currently empty
* `datasets/slapping.csv` - currently empty
* `datasets/generated_dataset.csv`

Each one of these datasets can be opened up this way
```python
import pandas as pd
from kiki.detection import get_resource_path

file = get_resource_path('datasets/generated_dataset.csv')
df = pd.read_csv(file)
```

`generated_dataset.csv` is a special dataset which was artificially built
from all the other non-empty datasets. Practically, it was generated
by iteratively altering the values from each sample by up to 15%. This should
provide enough variation to prove the point. All samples are shuffled. There are
 34000 samples in total, each one of them having 6 features and 11 time steps. 
 [kiki/datasets/data_generator.py](kiki/datasets/data_generator.py) was used to generate
 all this data. This script is not included in the package though.
 
 ## Pre-trained Model

The pre-trained model is found at `models/touch` and can be loaded by running 
```python
from kiki.detection import load_pretrained_model

load_pretrained_model('models/touch', inpackage_data=True)
```
This both loads the model of the neural network and the weights associated with it.

The neural network is comprised of an LSTM layer of 32 units and a dense layer with the same
number of neurons as there are features - currently 6 of them. The model returned is a 
keras [model](https://keras.io/models/model/). The test accuracy is as of this moment at 100%,
which should be expected from a dataset that is almost ideal and has almost no noise.