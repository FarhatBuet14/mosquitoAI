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
from PIL import Image

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# split_dr = "splitted_data_mainDataset/"
# parts = ["train", "test", "val"]

# for part in parts:
#     files = os.listdir(split_dr + part)
#     files = pd.DataFrame(files)
#     files.to_csv(split_dr + part + ".csv")
    

#########################################    Start Codes    #########################################

# --- Extract Annotations
anno_file = pd.read_csv("data/all.csv")
specimens = list(anno_file["#filename"])


with open("data/all.json") as f:
    anno_file = json.load(f)

specimens = list(anno_file.keys())

splitted_data = pd.read_csv("splitted_data_mainDataset/splitted_data.csv")

train = []
test = []
val = []

for index in tqdm(range(0, len(specimens))):
    specimen = specimens[index]
    part = splitted_data["parts"][splitted_data[splitted_data["name"]==anno_file[specimen]["filename"]].index.values[0]]
    path = f'splitted_data_mainDataset/{part}/{anno_file[specimen]["filename"]}'
    img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
    # img = Image.open(path)
    # img = np.asarray(img)
    width = img.shape[1]
    height =img.shape[0]
    
    for i in range(0, len(anno_file[specimen]['regions'])):
        mark = anno_file[specimen]['regions'][str(i)]
        data = {}
        c, x, y, w, h = list(mark.values())[0].values()
        data['file_path'] = path
        data['file_name'] = anno_file[specimen]["filename"]
        data['width'] = width
        data['height'] = height
        data["x_min"] = int(round(x))
        data["y_min"] = int(round(y))
        if(round(x+w) > width): data["x_max"] = int(round(width))
        else: data["x_max"] = int(round(x+w))
        if(round(y+h) > height): data["y_max"] = int(round(height))
        else: data["y_max"] = int(round(y+h))
        data['class_name'] = "larvae"
            
        # img_2 = annotate_image(img, [[x, y, h, w]])
        # cv2.imshow('ImageWindow', img_2)
        # cv2.waitKey()
        
        part = splitted_data["parts"][splitted_data[splitted_data["name"]==data['file_name']].index.values[0]]
        data["file_path"] = f'splitted_data_mainDataset/{part}/{anno_file[specimen]["filename"]}'
        if(part == "train"): train.append(data)
        elif(part == "test"): test.append(data)
        elif(part == "val"): val.append(data)


train_df = pd.DataFrame(train)
test_df = pd.DataFrame(test)
val_df = pd.DataFrame(val)

train_df.to_csv(f'annotation/train.csv', header=True, index=None)
val_df.to_csv(f'annotation/val.csv', header=True, index=None)
test_df.to_csv(f'annotation/test.csv', header=True, index=None)

print("Finished Prepartion...")