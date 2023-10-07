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
output_dir = f'output/'
anno_dir = "../../../data/working/annotation/larvae_localization"
test_dir = "../../../data/working/img/larvae_classification/classification_augmented_species_classification_4classes"

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
# dataset_dicts = create_dataset_dicts(train_df, classes)
# for d in random.sample(dataset_dicts, 3):
#     img = cv2.imread(d["file_name"])
#     visualizer = Visualizer(img[:, :, ::-1], metadata=statement_metadata)
#     vis = visualizer.draw_dataset_dict(d)
#     cv2.imshow("", vis.get_image()[:, :, ::-1])
#     cv2.waitKey()


# --- Set our own Trainer (add Evaluator for the test set )
class CocoTrainer(DefaultTrainer):
    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            os.makedirs("coco_eval", exist_ok=True)
            output_folder = "coco_eval"
        return COCOEvaluator(dataset_name, cfg, False, output_folder)


# --- Set the Configs
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
cfg.MODEL.DEVICE = 'cuda'
cfg.DATASETS.TRAIN = ("mark_train",)
cfg.DATASETS.TEST = ("mark_val",)

cfg.DATALOADER.NUM_WORKERS = 4
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")

cfg.SOLVER.IMS_PER_BATCH = 4
cfg.SOLVER.BASE_LR = 0.0125
cfg.SOLVER.MAX_ITER = 5000
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 256  
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 4
cfg.TEST.EVAL_PERIOD = 200
cfg.SOLVER.CHECKPOINT_PERIOD = 200

cfg.num_gpus = 2
cfg.output_dir = output_dir
os.makedirs(cfg.output_dir, exist_ok=True)
cfg.MODEL.MASK_ON = False

# --- Start Training
trainer = CocoTrainer(cfg) 
trainer.resume_or_load(resume=True)

# --- Evaluation Setup
cfg.MODEL.WEIGHTS = os.path.join(cfg.output_dir, "model_0002199.pth")
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.85
predictor = DefaultPredictor(cfg)

evaluator = COCOEvaluator("mark_val", cfg, False, output_dir=cfg.output_dir + "/validation/")
val_loader = build_detection_test_loader(cfg, "mark_val")
inference_on_dataset(trainer.model, val_loader, evaluator)

# ---- Test on Images
Annotation_folder = cfg.output_dir + "/annotated_results"
os.makedirs(Annotation_folder, exist_ok=True)
test_image_paths = os.listdir(test_dir)
# paths = test_df.file_path
# bbox = []
# for i in range(0, len(test_df)):
#     bbox.append([test_df.iloc[i]["x_min"] , test_df.iloc[i]["y_min"], test_df.iloc[i]["height"], test_df.iloc[i]["width"]])
# bbox = np.array(bbox)

for i in tqdm(range(0, len(test_image_paths))):
    clothing_image = test_image_paths[i]
    if(clothing_image != ".DS_Store"):
        im = cv2.imread(os.path.join(test_dir, clothing_image))
        outputs = predictor(im)
        
        instances = outputs["instances"].to("cpu")

        boxes = instances.pred_boxes if instances.has("pred_boxes") else None
        boxes = boxes.tensor.detach().numpy()
        classes = instances.pred_classes.tolist() if instances.has("pred_classes") else None

        for box, cls in zip(boxes, classes):

            box = [int(b) for b in box]
            crop_img = im[box[1]:box[3], box[0]:box[2]].copy()
            class_name = statement_metadata.thing_classes[cls]
            cv2.imwrite(f'./boxes/{class_name}/{clothing_image}.jpg', crop_img)

print("Finished Validation...")
