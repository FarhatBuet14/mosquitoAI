import pandas as pd
from urllib import request
import os
from tqdm import tqdm

folder_link = f'adultNET/compass'
df = pd.read_csv(f'{folder_link}/species.csv')

links = list(df.iloc[: ,0])
species = list(df.iloc[: ,1])
prob = list(df.iloc[: ,2])

for pos in tqdm(range(0, len(links))):
    i = links[pos]
    name = f'{pos}_{species[pos]}_{prob[pos]}'
    f = open(f'{folder_link}/input/{name}.jpg','wb')
    f.write(request.urlopen(i).read())
    f.close()