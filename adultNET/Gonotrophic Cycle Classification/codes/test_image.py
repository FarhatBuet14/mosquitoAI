import numpy as np
np.random.seed(9)
from keras.models import load_model
import cv2
from keras.preprocessing import image
import grad_cam
import argparse
import tensorflow as tf
import keras
import os

class paramClass():
    def __init__(self):
        self.model_name = "EfficientNetB0"
        self.model_directory = "../models/EfficientNetB0/model00000533.h5"
        self.image_name = "test_image.jpg"
param = paramClass()

parser = argparse.ArgumentParser(description='Necessary variables')
parser.add_argument("--name",type=int, help = "Name of the model architecture")
parser.add_argument("--model",type=int, help = "Directory to the pretrained model file")
parser.add_argument("--test",type=int, help = "Name of the testing image")
arguments = parser.parse_args()

if(arguments.name): param.model_name = arguments.name
if(arguments.model): param.model_directory = arguments.model
if(arguments.test): param.image_name = arguments.test

adult = {0: 'unfed', 1: 'gravid', 2: "semi-gravid", 3: "fully fed"}

if param.name == "ConvNeXtTiny":
    import keras.applications.convnext as cvx
    model = tf.keras.models.load_model(param.model_directory, compile = False, custom_objects={'LayerScale':cvx.LayerScale})
else:
    model = tf.keras.models.load_model(param.model_directory, compile = False)

optimizer = tf.keras.optimizers.Adam(learning_rate = 0.00001)
loss=keras.losses.SparseCategoricalCrossentropy()
model.compile(loss=loss,optimizer=optimizer,metrics=['accuracy', keras.metrics.SparseCategoricalAccuracy()])


img = cv2.resize(cv2.imread(param.image_name), (224, 224))
x = image.img_to_array(img) / 255
res = model.predict(np.array([x]).astype('float32')).tolist()[0]
predicted = adult[res.index(max(res))]
percentage = str(max(res) * 100)

print(param.image_name + " is " + predicted + " with a probability of " + percentage + "%") 

img = cv2.imread(os.path.join("random_test" , param.image_name))
height, width, _ = img.shape

superimposed_img = grad_cam.get_cam(model, x, height, width, threshold = 0)

alpha = 0.5
superimposed_img = superimposed_img * alpha + img
cv2.imwrite(f"gradCam_test_image", superimposed_img)
