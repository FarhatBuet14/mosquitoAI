import json
from urllib import request
import os
from tqdm import tqdm

data_link = f'data/FirstOne_HabitatMapper'
json_link = f'{data_link}/json_files'

with open(f'{json_link}/NASA.json') as f:
    data = json.load(f)
data = data["NASA_FILE_1"]
data_keys = [i for i in data.keys()]

for pos in tqdm(range(0, len(data_keys))):
    i = data_keys[pos]
    info = data[i]
    types = ["CLOSE_UP", "FULL_BODY", "WATER_SOURCES"]
    
    if(not os.path.isdir(f'{data_link}/data/{i}/')): os.makedirs(f'{data_link}/data/{i}/')
    name = [key for key in info["FULL_BODY"].keys()][0]
    
    for typo in types:

        if(not os.path.isdir(f'{data_link}/data/{i}/{typo}/')): os.makedirs(f'{data_link}/data/{i}/{typo}/')
        
        if(info[typo] == [""]): continue
        else:
            if(typo == "FULL_BODY"):
                j = info[typo][name]
                f = open(f'{data_link}/data/{i}/{typo}/{name}.jpg','wb')
                f.write(request.urlopen(j).read())
                f.close()
            else:
                for (num, j) in enumerate(info[typo]):
                    f = open(f'{data_link}/data/{i}/{typo}/{name}_{typo}_{num}.jpg','wb')
                    f.write(request.urlopen(j).read())
                    f.close()
