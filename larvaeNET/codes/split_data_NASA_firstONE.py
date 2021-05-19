import os
from tqdm import tqdm
import shutil
import json

data_link = f'data/FirstOne_HabitatMapper'
files = sorted(os.listdir(f'{data_link}/images/'))

grp = len(files) // 500
if(not os.path.isdir(f'{data_link}/images_gprs/')): os.makedirs(f'{data_link}/images_gprs/')
count = 0
dic = {}


for i in tqdm(range(0, grp)):
    for j in range((i*500)+1, (i+1)*500):
        if(not os.path.isdir(f'{data_link}/images_gprs/{i}/')): os.makedirs(f'{data_link}/images_gprs/{i}/')
        shutil.copy(f'{data_link}/images/{files[j]}', f'{data_link}/images_gprs/{i}/{files[j]}')
        count += 1
        dic.update({str(count) :
                {'name' : files[j],
                'group' : str(i), 
                'path' : f'{data_link}/images_gprs/{i}/{files[j]}'}})
    if(i==(grp-1)):
        for j in range((i+1)*500, len(files)):
            if(not os.path.isdir(f'{data_link}/images_gprs/{i+1}/')): os.makedirs(f'{data_link}/images_gprs/{i+1}/')
            shutil.copy(f'{data_link}/images/{files[j]}', f'{data_link}/images_gprs/{i+1}/{files[j]}')
            count += 1
            dic.update({str(count) :
                {'name' : files[j],
                'group' : str(i+1), 
                'path' : f'{data_link}/images_gprs/{i+1}/{files[j]}'}})

with open('{data_link}/json_files/NASA_raw_data__splitted_details.json', 'w') as json_file:
    json.dump(dic, json_file)
