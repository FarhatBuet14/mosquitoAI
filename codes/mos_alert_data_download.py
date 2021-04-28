# https://api-data.cscloud.host/mosquitoes/hourly?Source=iNaturalist&Topic=Mosquitoes&Format=Flat&Subscription-Key=97713c1cf12544d2804628c8b645a304


# import http.client, urllib.request, urllib.parse, urllib.error, base64

# headers = {
# # Request headers
# 'Cache-Control': 'no-cache',
# 'Ocp-Apim-Subscription-Key': '97713c1cf12544d2804628c8b645a304',
# }

# params = urllib.parse.urlencode({
# })

# try:
#     conn.request("GET", "?%s" % params, headers)
#     response = conn.getresponse()
#     data = response.read()
#     print(data)
#     conn.close()
# except Exception as e:
#     print("[Errno {0}] {1}".format(e.errno, e.strerror))


import pandas as pd 
from urllib import request
import os
from tqdm import tqdm

data = pd.read_csv("iNaturalist/" + "daily.csv")

url_list = list(data["observation_image_imageResult"].dropna())

for pos in tqdm(range(0, len(url_list))):
    i = url_list[pos]
    if(i):
        f = open(f'iNaturalist/download_daily/{pos}.jpg','wb')
        f.write(request.urlopen(i).read())
        f.close()


print("Finished")