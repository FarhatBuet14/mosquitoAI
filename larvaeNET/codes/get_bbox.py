import os
from tqdm import tqdm
import shutil
import json

image_dir = "data/larvae_photos/"
anno_dir = "larvaeNET/annotation/"

folder_names = {"culex": "Culex - 1519", "aedes": "Aedes - 2243", "anno": "Anopheles - 1672"}
species = list(folder_names.keys())
if(not os.path.isdir("larvaeNET/bbox/")): os.makedirs("larvaeNET/bbox/")

for spec in species:
    spec_dir = image_dir + folder_names[spec] + "/"

    if(not os.path.isdir("larvaeNET/bbox/confusion/")): os.makedirs("larvaeNET/bbox/confusion/")
    conf_dir = "larvaeNET/bbox/confusion/" + f'/{spec}/'
    if(not os.path.isdir(conf_dir)): os.makedirs(conf_dir)

    if(not os.path.isdir("larvaeNET/bbox/annotated_images/")): os.makedirs("larvaeNET/bbox/annotated_images/")
    annoImg_dir = "larvaeNET/bbox/annotated_images" + f'/{spec}/'
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
                values["file_path"] = annoImg_dir + name
                dic[images[i]] = values
        # else:
        #     shutil.copy(spec_dir + name, leftImg_dir + name)
    
    with open("larvaeNET/bbox/" + spec + '.json', 'w') as fp:
        json.dump(dic, fp)
    dic = {}

print("finished..")