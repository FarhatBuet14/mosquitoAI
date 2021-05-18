import os
from tqdm import tqdm
import shutil
import json

files = sorted(os.listdir(f'images/'))

grp = len(files) // 500
if(not os.path.isdir(f'images_gprs/')): os.makedirs(f'images_gprs/')
count = 0
dic = {}


for i in tqdm(range(0, grp)):
    for j in range((i*500)+1, (i+1)*500):
        if(not os.path.isdir(f'images_gprs/{i}/')): os.makedirs(f'images_gprs/{i}/')
        shutil.copy(f'images/{files[j]}', f'images_gprs/{i}/{files[j]}')
        count += 1
        dic.update({str(count) :
                {'name' : files[j],
                'group' : str(i), 
                'path' : f'images_gprs/{i}/{files[j]}'}})
    if(i==(grp-1)):
        for j in range((i+1)*500, len(files)):
            if(not os.path.isdir(f'images_gprs/{i+1}/')): os.makedirs(f'images_gprs/{i+1}/')
            shutil.copy(f'images/{files[j]}', f'images_gprs/{i+1}/{files[j]}')
            count += 1
            dic.update({str(count) :
                {'name' : files[j],
                'group' : str(i+1), 
                'path' : f'images_gprs/{i+1}/{files[j]}'}})

with open('json_files/NASA_raw_data__splitted_details.json', 'w') as json_file:
    json.dump(dic, json_file)
