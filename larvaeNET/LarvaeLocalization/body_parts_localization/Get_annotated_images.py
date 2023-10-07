import torch, torchvision
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()
import glob
import os
import ntpath
import numpy as np
import cv2
import random
import itertools
import pandas as pd
from tqdm import tqdm
import urllib
import json
import PIL.Image as Image
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor, DefaultTrainer
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.data import DatasetCatalog, MetadataCatalog, build_detection_test_loader
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.structures import BoxMode
import seaborn as sns
from pylab import rcParams
import matplotlib.pyplot as plt
from matplotlib import rc
from datetime import datetime


sns.set(style='whitegrid', palette='muted', font_scale=1.2)
HAPPY_COLORS_PALETTE = ["#01BEFE", "#FFDD00", "#FF7D00", "#FF006D", "#ADFF02", "#8F00FF"]
sns.set_palette(sns.color_palette(HAPPY_COLORS_PALETTE))
rcParams['figure.figsize'] = 12, 8
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

#########################################    Start Codes    #########################################

# --- Directories
directory = f'../../..'
data_dir = f'{directory}/data/working/img/larvae_localization'
anno_dir = f'{directory}/data/working/annotation/larvae_localization'
# output_dir = f'{directory}/data/working/annotation/larvae_localization_output'

# --- Load Annotations
train_df = pd.read_csv(f'{anno_dir}/train.csv')
val_df = pd.read_csv(f'{anno_dir}/val.csv')
# test_df = pd.read_csv(f'{anno_dir}/test.csv')

classes = train_df.class_name.unique().tolist()

# --- Dataset Dictionary for Training
def create_dataset_dicts(df, classes):
    dataset_dicts = []
    for image_id, img_name in enumerate(df.file_name.unique()):
        record = {}
        image_df = df[df.file_name == img_name]
        record["file_name"] = list(df[df.file_name == img_name]["file_path"])[0]
        record["file_path"] = list(df[df.file_name == img_name]["file_path"])[0]
        record["image_id"] = image_id
        record["height"] = int(image_df.iloc[0].height)
        record["width"] = int(image_df.iloc[0].width)
        objs = []
        for _, row in image_df.iterrows():
            xmin = int(row.x_min)
            ymin = int(row.y_min)
            xmax = int(row.x_max)
            ymax = int(row.y_max)
            obj = {
            "bbox": [xmin, ymin, xmax, ymax],
            "bbox_mode": BoxMode.XYXY_ABS,
            "category_id": classes.index(row.class_name),
            "iscrowd": 0
            }
            objs.append(obj)
        record["annotations"] = objs
        dataset_dicts.append(record)
    return dataset_dicts


# --- Assign the Dictionary
for d in ["train", "val"]:
    DatasetCatalog.register("mark_" + d, lambda d=d: create_dataset_dicts(train_df if d == "train" else val_df, classes))
    MetadataCatalog.get("mark_" + d).set(thing_classes=classes)
statement_metadata = MetadataCatalog.get("mark_train")


# --- Visualizing the Train Dataset Dictionary
dataset_dicts = create_dataset_dicts(val_df, classes)
for d in dataset_dicts:
    img = cv2.imread(d["file_name"])
    visualizer = Visualizer(img[:, :, ::-1], metadata=statement_metadata)
    vis = visualizer.draw_dataset_dict(d)
    cv2.imwrite(d["file_name"].replace("/larvae_localization/", "/larvae_localization_output/"), vis.get_image()[:, :, ::-1])
    # cv2.imshow("", vis.get_image()[:, :, ::-1])
    # cv2.waitKey()

print("Finished")