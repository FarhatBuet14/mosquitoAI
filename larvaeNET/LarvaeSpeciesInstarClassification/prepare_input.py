import cv2
import numpy as np
import pandas as pd
import shutil
from tqdm import tqdm

anno_folder = "data"

train_df = pd.read_csv(f'{anno_folder}/train_aug_16Class_allData_fixedCropError.csv')
val_df = pd.read_csv(f'{anno_folder}/val_16Class_allData_fixedCropError.csv')
test_df = pd.read_csv(f'{anno_folder}/test_16Class_allData_fixedCropError.csv')

larvae = {'anopheles_gambiae_3rd': 0, 'anopheles_gambiae_4th': 1, 'anopheles_stephensi_3rd': 2, 'anopheles_stephensi_4th': 3, 
          'anopheles_arabiensis_3rd': 4, 'anopheles_arabiensis_4th': 5, 'anopheles_funestus_3rd': 6, 'anopheles_funestus_4th': 7, 
          'aedes_albopictus_3rd': 8, 'aedes_albopictus_4th': 9, 'culex_quinquefasciatus_3rd': 10, 'culex_quinquefasciatus_4th': 11, 
          'aedes_aegypti_3rd': 12, 'aedes_aegypti_4th': 13, 'culex_tarsalis_3rd': 14, 'culex_tarsalis_4th': 15}

x_train = []
y_train = []
x_val = []
y_val = []
x_test = []
y_test = []

print("train data preparing,,,,")

for i, file in enumerate(tqdm(train_df['file_path'])):
    x_train.append(np.array(cv2.resize(cv2.imread(file), (224, 224))))
    y_train.append(np.array([larvae[train_df["class_name"][i]]]))

x_train = np.array(x_train)
y_train = np.array(y_train) 

assert x_train.shape == (len(y_train), 224, 224, 3)
assert y_train.shape == (len(y_train), 1)


print("validation data preparing,,,,")

for i, file in enumerate(tqdm(val_df['file_path'])):
    x_val.append(np.array(cv2.resize(cv2.imread(file), (224, 224))))
    y_val.append(np.array([larvae[val_df["class_name"][i]]]))

x_val = np.array(x_val)
y_val = np.array(y_val)

assert x_val.shape == (len(y_val), 224, 224, 3)
assert y_val.shape == (len(y_val), 1)


print("test data preparing,,,,")

for i, file in enumerate(tqdm(test_df['file_path'])):
    x_test.append(np.array(cv2.resize(cv2.imread(file), (224, 224))))
    y_test.append(np.array([larvae[test_df["class_name"][i]]]))

x_test = np.array(x_test)
y_test = np.array(y_test)

assert x_test.shape == (len(y_test), 224, 224, 3)
assert y_test.shape == (len(y_test), 1)

np.savez('data/data_16class_fixedCropError.npz', x_train=x_train, x_val=x_val, y_train=y_train, y_val=y_val, x_test=x_test, y_test=y_test)

print("Finished")