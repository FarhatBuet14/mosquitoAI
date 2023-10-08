import os
import shutil
import cv2
import numpy as np
import pandas as pd
import shutil

train_df = pd.read_csv(f'data/train_aug.csv')
val_df = pd.read_csv(f'data/val.csv')
test_df = pd.read_csv(f'data/test.csv')

adult = {'female': 0, 'male': 1}

x_train = []
y_train = []
x_val = []
y_val = []
x_test = []
y_test = []

for i, file in enumerate(train_df['file_path']):
    x_train.append(np.array(cv2.resize(cv2.imread(file), (224, 224))))
    y_train.append(np.array([adult[train_df["class_name"][i]]]))

x_train = np.array(x_train)
y_train = np.array(y_train)

assert x_train.shape == (864, 224, 224, 3) # 2184 - without manual crop
assert y_train.shape == (864, 1)


for i, file in enumerate(val_df['file_path']):
    x_val.append(np.array(cv2.resize(cv2.imread(file), (224, 224))))
    y_val.append(np.array([adult[val_df["class_name"][i]]]))

x_val = np.array(x_val)
y_val = np.array(y_val)

assert x_val.shape == (36, 224, 224, 3)
assert y_val.shape == (36, 1)


for i, file in enumerate(test_df['file_path']):
    x_test.append(np.array(cv2.resize(cv2.imread(file), (224, 224))))
    y_test.append(np.array([adult[test_df["class_name"][i]]]))

x_test = np.array(x_test)
y_test = np.array(y_test)

assert x_test.shape == (36, 224, 224, 3)
assert y_test.shape == (36, 1)

np.savez('data/data.npz', x_train=x_train, x_val=x_val, y_train=y_train, y_val=y_val, x_test=x_test, y_test=y_test)

print("Finished")