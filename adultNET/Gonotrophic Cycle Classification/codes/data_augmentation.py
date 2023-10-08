import pandas as pd
import os
import cv2
from tqdm import tqdm
import albumentations
import albumentations.pytorch

albumentations_transform = albumentations.Compose([
    albumentations.Rotate(limit=(30, 60), always_apply=True),
    albumentations.Rotate(limit=(-60, -30), always_apply=True),
    albumentations.VerticalFlip(p=1, always_apply=True),
    albumentations.HorizontalFlip(p=1, always_apply=True),
    albumentations.Sharpen(alpha=(0.5, 1.0), lightness=(1.0, 1.0), p=1, always_apply=True),
    albumentations.MedianBlur(blur_limit=[31, 51], p=1, always_apply=True),
    albumentations.RandomBrightnessContrast(brightness_limit=[-0.2, 0.1], contrast_limit=[-0.2, 0.1], p=1, always_apply=True)
])

data = pd.read_csv(f'{anno_folder}/train.csv')

aug_names = ["rotate_1", "rotate_2", "fliplr", "flipud", "sharp", "mediainBlur", "BrightnessContrast"]
dataset = []
for i, _ in tqdm(enumerate(data["file_name"])): 
    name = data["file_name"][i]
    img_path=data["file_path"][i]
    img = cv2.imread(img_path)

    aug_img = {}
    for bla in range(7):
        aug_img[aug_names[bla]] = albumentations_transform[bla](image = img)["image"]

    df=data[data['file_name']==name]

    for n in aug_names:
        spec_data = {}
        spec_data['file_name'] = f'{name.split(".jpg")[0]}_{n}.jpg'
        bla = img_path.split("/")
        bla[7] = 'adult_gonotrophy_4class_augmented/all_included'
        bla[8] = spec_data['file_name']
        directories = "/"
        for s in bla:
            directories = os.path.join(directories, s)
        spec_data['file_path'] = directories
        spec_data['class_name'] = data["class_name"][i]
        spec_data['specimen'] = data["specimen"][i]
        spec_data['genus_species'] = data["genus_species"][i]
        spec_data['sex'] = data["sex"][i]
        cv2.imwrite(spec_data["file_path"], aug_img[n])
        
        dataset.append(spec_data)

df = pd.DataFrame(dataset)

train_df = pd.concat([df, data])

train_df.to_csv(f'{anno_folder}/train_aug.csv', header=True, index=None)

print("Finished Augmnetation")
