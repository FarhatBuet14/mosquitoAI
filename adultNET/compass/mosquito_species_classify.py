import keras
from keras import applications
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers, regularizers
from keras.models import Sequential, Model
from keras.layers import Dropout, Flatten, Dense
from keras.models import load_model
from keras.preprocessing import image
import numpy as np
from keras.utils.generic_utils import CustomObjectScope
import time
import os
from shutil import copyfile
import tensorflow as tf
import os
from os import listdir
from os.path import isfile, join
import matplotlib.pyplot as plt
import itertools
import csv

mosquitoes = ['Aegypti', 'Crucians', 'Infirmatus', 'Nigrip', 'Peturbans', 'Taeniorh', 'Titillans']

#aedes aegypti, anopheless crucians, aedes Infirmatus, culex nigripalpus, coquillettidia perturbans, Aedes taeniorhynchus, mansonia titillans

model = load_model('0.8156weights.0.8025.hdf5')
file, input_path = "mosquito_march_18.jpg", "./input/"

#Uncomment below code to classify a single mosquito, change the file path above
"""
img = image.load_img(input_path + file, target_size=(299,299))
x = image.img_to_array(img) / 255
res = model.predict(np.array([x]).astype('float32')).tolist()[0]

print("This is " + mosquitoes[res.index(max(res))])

print("Probabilities are: \n")
for i, val in enumerate(res):
    print(mosquitoes[i] + " " + str(val * 100) + "%") 
"""
#Classify a folder of mosquitoes (comment out if the above is uncommented)
#The output is saved in csv file
X, filenames = [], []
for filename in os.listdir(input_path):
    if filename.endswith(".jpg") or filename.endswith(".jpeg") or filename.endswith(".png"):
        img = image.load_img(input_path + filename, target_size=(299,299))
        x = image.img_to_array(img) / 255
        X.append(x)
        filenames.append(filename)

with open('species.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    res = model.predict(np.array(X).astype('float32')).tolist()
    for i, val in enumerate(res):
        print(filenames[i] + " is " + mosquitoes[val.index(max(val))] + " with a probability of " + str(max(val)) + "%") 
        writer.writerow([filenames[i], mosquitoes[val.index(max(val))], '%.2f'%max(val)])