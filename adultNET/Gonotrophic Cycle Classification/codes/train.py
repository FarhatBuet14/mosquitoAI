import shutil
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.layers import AveragePooling2D, Flatten, Dense, Dropout, Concatenate, BatchNormalization, Input, Activation
from keras.models import Model
from keras.models import load_model
import argparse

config = tf.compat.v1.ConfigProto( device_count = {'GPU': 2 , 'CPU': 16} ) 
sess = tf.compat.v1.Session(config=config)
tf.compat.v1.keras.backend.set_session(sess)

data = np.load("../data/data.npz")

x_train = data["x_train"]
y_train = data["y_train"]
x_val = data["x_val"]
y_val = data["y_val"]

assert x_train.shape == (12032, 224, 224, 3)
assert y_train.shape == (12032, 1)
assert x_val.shape == (221, 224, 224, 3)
assert y_val.shape == (221, 1)

class paramClass():
    def __init__(self):
        self.model_name = "EfficientNetB0"
        self.epoch = 500
        self.batch_size = 16
        self.lr = 0.01
param = paramClass()

parser = argparse.ArgumentParser(description='Necessary variables')
parser.add_argument("--name",type=int, help = "Name of the model architecture")
parser.add_argument("--ep",type=int, help = "Number of epochs")
parser.add_argument("--batch",type=int, help = "Batch Size")
parser.add_argument("--lr",type=str, help = "Learning Rate")
arguments = parser.parse_args()

if(arguments.name): param.model_name = arguments.name
if(arguments.ep): param.epochs = arguments.ep
if(arguments.batch): param.batch_size = arguments.batch
if(arguments.lr): param.lr = arguments.lr


if param.model_name == "MobileNetV2": 
    baseModel = tf.keras.applications.MobileNetV2(
    input_shape=(224, 224, 3),
    alpha=0.75,
    include_top=False,
    weights="imagenet",
    input_tensor=Input(shape=(224, 224, 3)),
    pooling=None,
    classes=4)
    
elif param.model_name == "ConvNeXtTiny": 
    baseModel = tf.keras.applications.ConvNeXtTiny(
    model_name="convnext_tiny",
    include_top=False,
    include_preprocessing=True,
    weights="imagenet",
    input_tensor=Input(shape=(224, 224, 3)),
    input_shape=(224, 224, 3),
    pooling=None,
    classes=4)
    
elif param.model_name == "ResNet50": 
    baseModel = tf.keras.applications.ResNet50(
    include_top=False,
    weights="imagenet",
    input_tensor=Input(shape=(224, 224, 3)),
    input_shape=(224, 224, 3),
    pooling=None,
    classes=4)
    
else:
    baseModel = tf.keras.applications.EfficientNetB0(
    include_top=False, 
    input_tensor=Input(shape=(224, 224, 3)), 
    classes = 4)

baseModel.trainable = False

headModel = baseModel.output
headModel = AveragePooling2D(pool_size=(7, 7))(headModel) 
headModel = Flatten(name="flatten")(headModel)
headModel = Dense(256, activation="relu", kernel_initializer="glorot_uniform")(headModel)
headModel = BatchNormalization()(headModel)
headModel_1 = Dropout(0.5)(headModel)
headModel = Dense(128, activation="relu", kernel_initializer="glorot_uniform")(headModel_1)
headModel = BatchNormalization()(headModel)
headModel_2 = Dropout(0.5)(headModel)
headModel = Dense(128, activation="relu", kernel_initializer="glorot_uniform")(headModel_2)
headModel = BatchNormalization()(headModel)
headModel_3 = Dropout(0.5)(headModel)
headModel = Concatenate()([headModel_1, headModel_2, headModel_3])
headModel = Dense(4, activation="softmax")(headModel)

model = Model(inputs=baseModel.input, outputs=headModel)

# # Load model
# model = load_model("model00000328.h5")
# model.trainable = True

print(model.summary())

optimizer = tf.keras.optimizers.Adam(learning_rate = param.lr)
loss=keras.losses.SparseCategoricalCrossentropy()
model.compile(loss=loss,optimizer=optimizer,metrics=['accuracy', keras.metrics.SparseCategoricalAccuracy()])
checkpoint = keras.callbacks.ModelCheckpoint(log_folder + '../models/param.model_name/model{epoch:08d}.h5', save_freq=1)

from keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(rescale=1.0/255.0)
val_datagen = ImageDataGenerator(rescale=1.0/255.0)

train_generator = train_datagen.flow(x_train, y_train, batch_size=param.batch_size, shuffle=True)
val_generator = val_datagen.flow(x_val, y_val, batch_size=8, shuffle=True)

# fit model
history = model.fit(train_generator, steps_per_epoch=len(train_generator),
    validation_data=val_generator, validation_steps=len(val_generator), callbacks=[checkpoint], 
    epochs=param.epoch, shuffle=True, verbose=1)

hist_df = pd.DataFrame(history.history) 
hist_df.to_csv(log_folder + '../models/param.model_name/Final_history.csv')

print("Finished")
