import numpy as np
import pandas as pd
import os
import cv2
from tqdm import tqdm
from tensorflow.keras.utils import to_categorical

anno_folder = "../../../../data/working/annotation/classification"
main_data = pd.read_csv(f'../../../../data/working/img/larvae_classification/classification_35Spec__tarsalis_L3L4/classification_35Spec__tarsalis_L3L4.csv')

specimens = list(main_data["name"])
gs_classes = list(main_data["genus_species"])
paths = list(main_data["path"])

class_dict = {}

classes_list = ['anopheles_gambiae_3rd', 'anopheles_gambiae_4th', 'anopheles_stephensi_3rd', 'anopheles_stephensi_4th', \
                'anopheles_arabiensis_3rd', 'anopheles_arabiensis_4th', 'anopheles_funestus_3rd', 'anopheles_funestus_4th', \
                    'aedes_albopictus_3rd', 'aedes_albopictus_4th', 'culex_quinquefasciatus_3rd', 'culex_quinquefasciatus_4th', \
                        'aedes_aegypti_3rd', 'aedes_aegypti_4th', 'culex_tarsalis_3rd', 'culex_tarsalis_4th']

classes = []
for i, gs in enumerate(gs_classes):
    classes.append(f'{gs}_{list(main_data["instar"])[i]}')

for i, cls in enumerate(classes_list): class_dict[cls] = i
classes_int = [class_dict[cls] for cls in classes]
classes_int = list(to_categorical(classes_int, num_classes = len(classes_list)))

train_df = pd.DataFrame()
val_df = pd.DataFrame()
test_df = pd.DataFrame()

train_cls_df = pd.DataFrame()
val_cls_df = pd.DataFrame()
test_cls_df = pd.DataFrame()

dataset = []
cls_dataset = []
for index in tqdm(range(0, len(specimens))):
    specimen = specimens[index]
    data = {}
    data['file_path'] = paths[index]
    data['file_name'] = specimens[index]
    data['class_name'] = classes[index]
    data['specimen'] = specimens[index].split("_")[3][1:]
    data['source'] = list(main_data["img_source"])[index]
    data["source_name"] = list(main_data["source_name"])[index]
    cls = {}
    for i, c in enumerate(classes_int[index]):
        cls[str(i)] = c
    
    dataset.append(data)
    cls_dataset.append(cls)

df = pd.DataFrame(dataset)
cls_df = pd.DataFrame(cls_dataset)

# ------------------------- Train-Validation-Test Splitting ----------------------------

class_files = df.class_name.unique()
sources = df.source.unique()

# for s in range(len(sources)):
#     unique_df__ = df[df.source.isin([sources[s]])]
#     unique_cls_df__ = cls_df[df.source.isin(([sources[s]]))]
    
for i in range(len(class_files)):
    # unique_df = unique_df__[unique_df__.class_name.isin([class_files[i]])]
    # unique_cls_df = unique_cls_df__[unique_df__.class_name.isin(([class_files[i]]))]

    unique_df = df[df.class_name.isin([class_files[i]])]
    unique_cls_df = df[df.class_name.isin(([class_files[i]]))]

    unique_files = unique_df.specimen.unique()
    train_files = set(np.random.choice(unique_files, int(len(unique_files) * 0.80), replace=False))
    train_df = pd.concat([train_df, unique_df[unique_df.specimen.isin(train_files)]])
    train_cls_df = pd.concat([train_cls_df, unique_cls_df[unique_df.specimen.isin(train_files)]])
    rest = unique_df[~unique_df.specimen.isin(train_files)]
    cls_rest = unique_cls_df[~unique_df.specimen.isin(train_files)]

    unique_files = rest.specimen.unique()
    val_files = set(np.random.choice(unique_files, int(len(unique_files) * 0.50), replace=False))
    val_df = pd.concat([val_df, rest[rest.specimen.isin(val_files)]])
    val_cls_df = pd.concat([val_cls_df, cls_rest[rest.specimen.isin(val_files)]])

    test_df = pd.concat([test_df, rest[~rest.specimen.isin(val_files)]])
    test_cls_df = pd.concat([test_cls_df, cls_rest[~rest.specimen.isin(val_files)]])

train_df.to_csv(f'{anno_folder}/species_classification_16classes/train.csv', header=True, index=None)
val_df.to_csv(f'{anno_folder}/species_classification_16classes/val.csv', header=True, index=None)
test_df.to_csv(f'{anno_folder}/species_classification_16classes/test.csv', header=True, index=None)

train_cls_df.to_csv(f'{anno_folder}/species_classification_16classes/train_cls.csv', header=True, index=None)
val_cls_df.to_csv(f'{anno_folder}/species_classification_16classes/val_cls.csv', header=True, index=None)
test_cls_df.to_csv(f'{anno_folder}/species_classification_16classes/test_cls.csv', header=True, index=None)

print("Finished Prepartion...")
