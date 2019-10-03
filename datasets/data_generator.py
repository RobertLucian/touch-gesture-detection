import numpy as np
import pandas as pd
import random

# read all source datasets
source_files = [
    'double_tapping.csv',
    'tapping.csv',
    'horizontal_swipe.csv',
    'vertical_swipe.csv'
]
appended_data = []
for file in source_files:
    print(file)
    appended_data.append(pd.read_csv(file))
data = pd.concat(appended_data, sort=False)

# function to add variation to a reading
def add_variation(val):
    # induced variation within the dataset
    variation_percentage = 0.15 # 15%
    variation = 2 * variation_percentage * (random.random() - 0.5)
    # reduce the variation down to a quarter if the value is max
    if val == 4095:
        variation /= 5
    new_val = int(val * (variation + 1))
    if new_val < 0:
        new_val = 0
    if new_val > 4095:
        new_val = 4095
    return new_val

# set number of synthetic samples
samples = 2000
timestamps_per_sample = 11
samples_original = data.shape[0] // timestamps_per_sample
appended_synthetic_data = []

# generate [samples] samples from the existing dataset
for i in range(samples):
    for idx in range(samples_original):
        start = idx * timestamps_per_sample
        end = (idx + 1) * timestamps_per_sample
        new_data = data.iloc[start: end].copy()
        new_data.loc[:, 'A':'F'] = new_data.loc[:, 'A':'F'].applymap(add_variation)
        appended_synthetic_data.append(new_data)

# shuffle data and transform to dataframe
random.shuffle(appended_synthetic_data)
synthetic_data = pd.concat(appended_synthetic_data)

# and write it off to the disk
synthetic_data.to_csv('generated_dataset.csv', index=False)