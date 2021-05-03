# python prepare_input.py

import os
import numpy as np
import cv2
import random
import itertools
import pandas as pd
from tqdm import tqdm
import seaborn as sns
from pylab import rcParams
import json

sns.set(style='whitegrid', palette='muted', font_scale=1.2)
HAPPY_COLORS_PALETTE = ["#01BEFE", "#FFDD00", "#FF7D00", "#FF006D", "#ADFF02", "#8F00FF"]
sns.set_palette(sns.color_palette(HAPPY_COLORS_PALETTE))
rcParams['figure.figsize'] = 12, 8
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

#########################################    Start Codes    #########################################

# --- Directories
directory = f'/Users/farhat/farhat_files/google_backup/research/GitHub/mosquitoNET'
data_dir = f'{directory}/bbox/annotated_images'
anno_dir = f'{directory}/bbox'
species = {"aedes", "anno", "culex"}

# --- Test Annotations
# def annotate_image(pth, marks):
#     img = cv2.cvtColor(cv2.imread(pth), cv2.COLOR_BGR2RGB)
#     for mark in marks:
#         x, y, h, w = mark
#         cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
#     return img

# aList = [i for i in range(len(paths))]
# for i in random.sample(aList, 10):
#     img = annotate_image(paths[i], bbox[i])
#     cv2.imshow('ImageWindow', img)
#     cv2.waitKey()


# --- Extract Annotations
train_df = pd.DataFrame()
val_df = pd.DataFrame()
test_df = pd.DataFrame()

for spec in species:

    dataset = []
    
    with open(f'{anno_dir}/{spec}.json') as f:
        print(f'Reading {spec} Data')
        anno_file = json.load(f)

    specimens = list(anno_file.keys())

    for index in tqdm(range(0, len(specimens))):
        specimen = specimens[index]
        img = cv2.cvtColor(cv2.imread(f'{directory}/{anno_file[specimen]["file_path"]}'), cv2.COLOR_BGR2RGB)
        width = img.shape[1]
        height =img.shape[0]
        for i in range(0, len(anno_file[specimen]['regions'])):
            mark = anno_file[specimen]['regions'][str(i)]
            data = {}
            c, x, y, w, h = mark["shape_attributes"].values()
            data['file_path'] = anno_file[specimen]["file_path"]
            data['file_name'] = anno_file[specimen]["filename"]
            data['width'] = width
            data['height'] = height
            data["x_min"] = int(round(x))
            data["y_min"] = int(round(y))
            if(round(x+w) > width): data["x_max"] = int(round(width))
            else: data["x_max"] = int(round(x+w))
            if(round(y+h) > height): data["y_max"] = int(round(height))
            else: data["y_max"] = int(round(y+h))
            data['class_name'] = mark["region_attributes"]["bbox"]
            dataset.append(data)

    df = pd.DataFrame(dataset)

    # --- Train-Validation-Test Splitting
    unique_files = df.file_name.unique()
    train_files = set(np.random.choice(unique_files, int(len(unique_files) * 0.50), replace=False))
    train_df = pd.concat([train_df, df[df.file_name.isin(train_files)]])
    rest = df[~df.file_name.isin(train_files)]
    
    unique_files = rest.file_name.unique()
    val_files = set(np.random.choice(unique_files, int(len(unique_files) * 0.50), replace=False))
    val_df = pd.concat([val_df, rest[rest.file_name.isin(val_files)]])
    test_df = pd.concat([test_df, rest[~rest.file_name.isin(val_files)]])

train_df.to_csv(f'codes/fbc_det2/train.csv', header=True, index=None)
val_df.to_csv(f'/codes/fbc_det2/val.csv', header=True, index=None)
test_df.to_csv(f'/codes/fbc_det2/test.csv', header=True, index=None)

print("Finished Prepartion...")