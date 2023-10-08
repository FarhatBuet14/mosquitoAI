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
import shutil

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

#########################################    Start Codes    #########################################

# --- Directories
directory = f'/Users/farhat/farhat_files/google_backup/research/GitHub/mosquitoAI/data/larvae_photos/Volunteers_Annotation_Folder'
data_dir = f'{directory}/data_distribution'
anno_dir = f'{directory}/annotation'
move_dir = "../../../data/working/"

# --- Extract Annotations
train_df = pd.DataFrame()
val_df = pd.DataFrame()
test_df = pd.DataFrame()


annotations = os.listdir(anno_dir)

for anno in annotations:

    dataset = []

    with open(f'{anno_dir}/{anno}') as f:
        print(f'Reading {anno} Data')
        anno_file = json.load(f)

    specimens = list(anno_file.keys())

    bla = []
    for s in specimens:
        if("(1)" not in s): bla.append(s)
    specimens = bla

    for index in tqdm(range(0, len(specimens))):
        specimen = specimens[index]
        path = f'{data_dir}/{anno.split(".")[0]}/{anno_file[specimen]["filename"]}'
        move_path = f'{move_dir}/img/larvae_photos/{anno_file[specimen]["filename"]}'
        img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
        shutil.copy(path, move_path)
        width = img.shape[1]
        height =img.shape[0]
        for i in range(0, len(anno_file[specimen]['regions'])):
            mark = anno_file[specimen]['regions'][str(i)]
            data = {}
            c, x, y, w, h = mark["shape_attributes"].values()
            data['file_path'] = move_path
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
            
            # img_2 = annotate_image(img, [[x, y, h, w]])
            # cv2.imshow('ImageWindow', img_2)
            # cv2.waitKey()
            
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

train_df.to_csv(f'{move_dir}/annotation/new/train.csv', header=True, index=None)
val_df.to_csv(f'{move_dir}/annotation/new/val.csv', header=True, index=None)
test_df.to_csv(f'{move_dir}/annotation/new/test.csv', header=True, index=None)

print("Finished Prepartion...")