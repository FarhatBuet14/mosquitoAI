import pandas as pd 
from urllib import request
import os
from tqdm import tqdm
import numpy as np 

# ---------------  iNaturalist  -----------------
# directory = "data/portal_data/iNaturalist"
# data = pd.read_csv(f'{directory}/daily.csv').replace(np.nan, 0)

# url_list = list(data["observation_image_imageResult"])
# ids = list(data["observation_projectObservationUID"])
# timestamps = list(data["ETL_TIMESTAMP"])

# if(not os.path.isdir(f'{directory}/download_daily/')): os.makedirs(f'{directory}/download_daily/')

# for pos in tqdm(range(0, len(url_list))):
#     if(url_list[pos] != 0):
#         f = open(f'{directory}/download_daily/{ids[pos]}_{timestamps[pos]}.jpg','wb')
#         f.write(request.urlopen(url_list[pos]).read())
#         f.close()

# ---------------  mosquitoAlert  -----------------
# directory = "data/portal_data/mosquitoAlert"
# data = pd.read_csv(f'{directory}/di_mafm_mosquitoAlert_mosquitoes_2021_derived.csv').replace(np.nan, 0)

# url_list = list(data["observation.image.imageResult"])
# ids = list(data["observation.projectObservationUID"])
# timestamps = list(data["observation.resultCategoryObservation.submitTime"])

# if(not os.path.isdir(f'{directory}/download_daily/')): os.makedirs(f'{directory}/download_daily/')

# for pos in tqdm(range(0, len(url_list))):
#     if(url_list[pos] != 0):
#         f = open(f'{directory}/download_daily/{ids[pos]}_ \
#             {timestamps[pos].replace(":", "-").replace(".", "=")}.jpg','wb')
#         f.write(request.urlopen(url_list[pos]).read())
#         f.close()

# import http.client, urllib.request, urllib.parse, urllib.error, base64

# headers = {
# # Request headers
# 'Cache-Control': 'no-cache',
# 'Ocp-Apim-Subscription-Key': '97713c1cf12544d2804628c8b645a304',
# }

# params = urllib.parse.urlencode({
# })

# conn = http.client.HTTPSConnection('api-data.cscloud.host/mosquitoes/daily')
# conn.request("GET", "?%s" % params, headers)
# response = conn.getresponse()
# data = response.read()
# print(data)
# conn.close()

print("Finished")