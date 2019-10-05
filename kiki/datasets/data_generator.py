import pandas as pd
import numpy as np
from tqdm import tqdm
import random

# read all source datasets
source_files = [
    'double_tapping.csv',
    'tapping.csv',
    'horizontal_swipe.csv',
    'vertical_swipe.csv',
    'slapping.csv'
]

print("Aggregating datasets")
appended_data = []
for file in source_files:
    print(f"Adding '{file}' dataset to pool")
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
timesteps = 11
samples_original = data.shape[0] // timesteps
appended_synthetic_data = []

# generate [samples] samples from the existing dataset
print("Shuffling and adding variation while creating the synthetic dataset")
for i in tqdm(range(samples)):
    for idx in range(samples_original):
        start = idx * timesteps
        end = (idx + 1) * timesteps
        new_data = data.iloc[start: end].copy()
        new_data.loc[:, 'A':'F'] = new_data.loc[:, 'A':'F'].applymap(add_variation)
        appended_synthetic_data.append(new_data)

# shuffle data and transform to dataframe
random.shuffle(appended_synthetic_data)
synthetic_data = pd.concat(appended_synthetic_data)

# select candidate indexes that are not divisible by [timesteps]
no_choice_samples = 2200
rows = samples * timesteps * samples_original
indexes = np.arange(rows)
indexes = indexes[indexes % timesteps != 0]
indexes = np.random.choice(indexes, no_choice_samples, replace=False)

# create dataframes with indexes that don't start
# at indexes divisible by [timesteps]
print("Adding samples that map to no action")
for i in tqdm(range(no_choice_samples)):
    start = indexes[i]
    end = start + timesteps
    window = synthetic_data.iloc[start:end].copy()
    appended_synthetic_data.append(window)

# shuffle data and transform to dataframe
# this time including samples with no detected gesture
random.shuffle(appended_synthetic_data)
synthetic_data = pd.concat(appended_synthetic_data)

# and write it off to the disk
print("Saving the generated dataset to disk")
synthetic_data.to_csv('generated_dataset.csv', index=False)