from sklearn.model_selection import train_test_split
from collections import Counter
from tqdm import tqdm

import pandas as pd
import numpy as np
import random
import math
import os

from Utility import log
import config

train_test_split_percentage = 0.8
bins = 50
seed = 0

training_path = os.path.join(config.BASE_DIR, 'sampled_training.csv')
testing_path = os.path.join(config.BASE_DIR, 'unsampled_testing.csv')

log.info('Reading data')
df = pd.read_csv(config.TRAIN_PATH, header=0, usecols=['target', 'comment_text'])
training_df, testing_df = train_test_split(df, test_size = 1.0 - train_test_split_percentage, random_state=seed)

log.info('Saving testing data')
f = open(testing_path, 'w')
f.write(testing_df.to_csv())
f.close()
log.info(f'Saved testing data to: {testing_path}')


# Putting data into bins
x = {}
y = {}

for val in tqdm(df.iterrows(), desc='collecting data into bins'):
    target = val[1][0]
    bin_index = math.floor(target / (1/bins))
    
    if bin_index not in x:
        x[bin_index] = []
        y[bin_index] = []
        
    x[bin_index].append(val[1][1])
    y[bin_index].append(target)


# Now we find the bin with the second most occurrences and the associated count.
counter = Counter()

for key in tqdm(y, desc='counting bins'):
    dictionary = {}
    dictionary[key] = len(y[key])
    counter.update(Counter(dictionary))

most_common = counter.most_common(2)
second_most_common_count = most_common[1][1]

most_common_key = most_common[0][0]
most_common_count = most_common[0][1]

random.seed(seed)
indexes = [i for i in range(most_common_count)]
indexes = random.sample(indexes, second_most_common_count)

new_x = []
new_y = []

for key in tqdm(y, desc='reconstructing set based on samples'):
    if key == most_common_key:
        for index in indexes:
            new_x.append(x[key][index])
            new_y.append(y[key][index])
    else:
        new_x.extend(x[key])
        new_y.extend(y[key])

# Now we can create the dataframe and see the results of our hardwork undersampling.
log.info('constructing dataframe')
data = {'target': new_y, 'comment_text': new_x}
df = pd.DataFrame(data)

log.info('Saving training data')
f = open(training_path, 'w')
f.write(df.to_csv())
f.close()
log.info(f'Saved training data to: {training_path}')