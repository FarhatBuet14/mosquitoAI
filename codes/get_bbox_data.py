import os
from tqdm import tqdm
import shutil
import json

image_dir = "larvae_photos/"
anno_dir = "annotation/"

folder_names = {"culex": "Culex - 1519", "aedes": "Aedes - 2243", "anno": "Anopheles - 1672"}
species = list(folder_names.keys())
if(not os.path.isdir("bbox/")): os.makedirs("bbox/")

for spec in species:
    # spec = species[0]
    spec_dir = image_dir + folder_names[spec] + "/"

    if(not os.path.isdir("bbox/confusion/")): os.makedirs("bbox/confusion/")
    conf_dir = "bbox/confusion/" + f'/{spec}/'
    if(not os.path.isdir(conf_dir)): os.makedirs(conf_dir)

    if(not os.path.isdir("bbox/annotated_images/")): os.makedirs("bbox/annotated_images/")
    annoImg_dir = "bbox/annotated_images/" + f'/{spec}/'
    if(not os.path.isdir(annoImg_dir)): os.makedirs(annoImg_dir)

    # if(not os.path.isdir("bbox/left_images/")): os.makedirs("bbox/left_images/")
    # leftImg_dir = "bbox/left_images/" + f'/{spec}/'
    # if(not os.path.isdir(leftImg_dir)): os.makedirs(leftImg_dir)

    with open(anno_dir + spec + ".json") as f:
        print(f'Reading {spec} Data')
        data = json.load(f)

    images = list(data.keys())
    dic = {}

    for i in tqdm(range(0, len(images))):
        values = data[images[i]]
        name = values["filename"]
        if(len(list(values['regions'].keys())) != 0):
            parts = [values["regions"][part]["region_attributes"]["bbox"] for part in list(values['regions'].keys())]
            if "confusion" in parts:
                shutil.copy(spec_dir + name, conf_dir + name)
            else:
                shutil.copy(spec_dir + name, annoImg_dir + name)
                dic[images[i]] = values
        # else:
        #     shutil.copy(spec_dir + name, leftImg_dir + name)
    
    with open("bbox/" + spec + '.json', 'w') as fp:
        json.dump(dic, fp)
    dic = {}

print("finished..")