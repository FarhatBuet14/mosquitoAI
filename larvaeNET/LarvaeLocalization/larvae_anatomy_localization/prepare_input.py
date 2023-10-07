import os
import shutil
import numpy as np
import cv2
import random
import itertools
import pandas as pd
from tqdm import tqdm
import seaborn as sns
from pylab import rcParams
import json
from keras.utils import to_categorical

sns.set(style='whitegrid', palette='muted', font_scale=1.2)
HAPPY_COLORS_PALETTE = ["#01BEFE", "#FFDD00", "#FF7D00", "#FF006D", "#ADFF02", "#8F00FF"]
sns.set_palette(sns.color_palette(HAPPY_COLORS_PALETTE))
rcParams['figure.figsize'] = 12, 8
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

#########################################    Start Codes    #########################################

# --- Directories
directory = f'../../..'
data_dir = f'{directory}/data/working/img/larvae_localization'
anno_dir = f'{directory}/data/working/annotation/larvae_localization'

# --- Extract Annotations

class_names = {"h": "head", "t": "thorax", "a": "abdomen", "l": "lower"}

data = pd.read_csv(f'{anno_dir}/localization_data.csv')
classes = list(data.iloc[:, 2].astype(str))

# --- Test Annotations
def annotate_image(img, marks):
    for mark in marks:
        x, y, h, w = mark
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
    return img

dataset = []
reject_list = []
with open(f'{anno_dir}/via_region_data.json') as f:
    print(f'Reading Data')
    anno_file = json.load(f)

specimens = list(anno_file.keys())

for index in tqdm(range(0, len(specimens))):
    specimen = specimens[index]
    source_path = f'{data_dir}/{anno_file[specimen]["filename"]}'
    img = cv2.cvtColor(cv2.imread(source_path), cv2.COLOR_BGR2RGB)
    width = img.shape[1]
    height =img.shape[0]
    if(len(anno_file[specimen]['regions']) == 0): 
        # reject_list.append(f'{data_dir}/{anno_file[specimen]["filename"]}')
        print("No box added")
        continue
    for i in range(0, len(anno_file[specimen]['regions'])):
        mark = anno_file[specimen]['regions'][str(i)]
        if(mark["region_attributes"] == {}):
            # reject_list.append(f'{data_dir}/{anno_file[specimen]["filename"]}')
            print("ERROR !!!!!!")
            continue
        if(mark["region_attributes"]["bbox"] == "no"):
            reject_list.append(f'{data_dir}/{anno_file[specimen]["filename"]}')
            print("box found but no region")
            continue
        shutil.copy(source_path, source_path.replace("/larvae_localization/", "/larvae_localization_rest/"))
        data = {}
        c, x, y, w, h = mark["shape_attributes"].values()
        data['file_path'] = f'{data_dir}/{anno_file[specimen]["filename"]}'
        data['file_name'] = anno_file[specimen]["filename"]
        data['specimen'] = int(anno_file[specimen]["filename"].split("_")[2][1:])
        data['width'] = width
        data['height'] = height
        data["x_min"] = int(round(x))
        data["y_min"] = int(round(y))
        if(round(x+w) > width): data["x_max"] = int(round(width))
        else: data["x_max"] = int(round(x+w))
        if(round(y+h) > height): data["y_max"] = int(round(height))
        else: data["y_max"] = int(round(y+h))
        data['class_name'] = class_names[mark["region_attributes"]["bbox"]]
        
        # img_2 = annotate_image(img, [[x, y, h, w]])
        # cv2.imshow('ImageWindow', img_2)
        # cv2.waitKey()
        
        dataset.append(data)

df = pd.DataFrame(dataset)

# --- Train-Validation-Test Splitting

# class_files = df.class_name.unique()

train_df = pd.DataFrame()
val_df = pd.DataFrame()

# for i in range(len(class_files)):
# df = df[df.class_name.isin([class_files[i]])]
unique_files = df.specimen.unique()
# print(f"{class_files[i]}  -- {sum(df.class_name.isin([class_files[i]]))}")

train_files = set(np.random.choice(unique_files, int(len(unique_files) * 0.75), replace=False))
train_df = pd.concat([train_df, df[df.specimen.isin(train_files)]])
rest = df[~df.specimen.isin(train_files)]

val_df = pd.concat([val_df, rest[~rest.specimen.isin(train_files)]])

train_df.to_csv(f'{anno_dir}/train.csv', header=True, index=None)
val_df.to_csv(f'{anno_dir}/val.csv', header=True, index=None)

# import shutil
# for path in reject_list: 
#     shutil.copy(path, path.replace("/larvae_localization/", "/larvae_localization_rejected/"))

print("Finished Prepartion...")
