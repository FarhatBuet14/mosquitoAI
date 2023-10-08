from keras.models import load_model
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import keras
import tensorflow as tf
import argparse

class paramClass():
    def __init__(self):
        self.model_name = "EfficientNetB0"
        self.model_directory = "../models/EfficientNetB0/model00000533.h5"
param = paramClass()

parser = argparse.ArgumentParser(description='Necessary variables')
parser.add_argument("--name",type=int, help = "Name of the model architecture")
parser.add_argument("--model",type=int, help = "Directory to the pretrained model file")
arguments = parser.parse_args()

if(arguments.name): param.model_name = arguments.name
if(arguments.model): param.model_directory = arguments.model

data = np.load("../data/data.npz")

x_test = data["x_test"]
y_test = data["y_test"]
assert x_test.shape == (234, 224, 224, 3)
assert y_test.shape == (234, 1)
x_test = x_test / 255.0


if param.model_name == "ConvNeXtTiny":
    import keras.applications.convnext as cvx
    model = tf.keras.models.load_model(param.model_directory, compile = False, custom_objects={'LayerScale':cvx.LayerScale})
else:
    model = tf.keras.models.load_model(param.model_directory, compile = False)

optimizer = tf.keras.optimizers.Adam(learning_rate = 0.00001)
loss=keras.losses.SparseCategoricalCrossentropy()
model.compile(loss=loss,optimizer=optimizer,metrics=['accuracy', keras.metrics.SparseCategoricalAccuracy()])

res = model.predict(x_test)
res_test = np.argmax(res, axis=1).reshape((234, 1))

cm_ = confusion_matrix(y_test,res_test)
accuracy_ = (sum(res_test == y_test) / len(y_test))[0] * 100
print(classification_report(y_test, res_test))
