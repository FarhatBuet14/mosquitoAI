import shutil
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.callbacks import CSVLogger
from keras.layers import GlobalAveragePooling2D, Flatten, Dense, Dropout, Concatenate, BatchNormalization, Input, Activation
from keras.models import Model
from datetime import datetime
from keras.models import load_model
from tensorflow.keras import optimizers
from tensorflow.keras import models
from tensorflow.keras import layers

config = tf.compat.v1.ConfigProto( device_count = {'GPU': 8 , 'CPU': 32} ) 
sess = tf.compat.v1.Session(config=config)
tf.compat.v1.keras.backend.set_session(sess)

data = np.load("data/data.npz")

x_train = data["x_train"]  
y_train = data["y_train"]
x_val = data["x_val"]
y_val = data["y_val"]
# x_test = data["x_test"]
# y_test = data["y_test"]

assert x_train.shape == (864, 224, 224, 3) # 2184 - without manual crop
assert y_train.shape == (864, 1)
assert x_val.shape == (36, 224, 224, 3)
assert y_val.shape == (36, 1)

log_folder = "logs/efficientNetB0/16_1e-5_adam_sparseLoss_2layers_1/"
shutil.copy("train_keras_eff.py", log_folder + "train_keras_eff.py")

baseModel = tf.keras.applications.EfficientNetB0(include_top=False, input_tensor=Input(shape=(224, 224, 3)), classes = 2)

# two Dense Layer Add
headModel = baseModel.output
headModel = GlobalAveragePooling2D()(headModel) 
# (7, 7) for resnet50, (5, 5) for inceptionResnetV2 # (12, 12) for 433 layer incresV2, #(7, 7) for efficientNetV2B0
headModel = Flatten(name="flatten")(headModel)
headModel = Dense(256, activation="relu", kernel_initializer="glorot_uniform")(headModel)
headModel = BatchNormalization()(headModel)
headModel_1 = Dropout(0.5)(headModel)
# headModel = Dense(128, activation="relu", kernel_initializer="glorot_uniform")(headModel_1)
# headModel = BatchNormalization()(headModel)
# headModel_2 = Dropout(0.5)(headModel)
headModel = Dense(64, activation="relu", kernel_initializer="glorot_uniform")(headModel_1)
headModel = BatchNormalization()(headModel)
headModel_3 = Dropout(0.5)(headModel)
# headModel = Concatenate()([headModel_1, headModel_2, headModel_3])
headModel = Dense(2, activation="softmax")(headModel_3)

model = Model(inputs=baseModel.input, outputs=headModel)

# # Load model
# model = load_model("logs/logs_balancedData_35SpecPerSpecies/efficientNetV2/16_0.001_1Layer_Dense_FullTraining/best/model00000581.h5")
# # model.trainable = True

model.summary()

# optimizer = optimizers.RMSprop(lr=2e-5)
loss=keras.losses.SparseCategoricalCrossentropy()
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-5)
model.compile(loss=loss,optimizer=optimizer,metrics=['accuracy', keras.metrics.SparseCategoricalAccuracy()])
checkpoint = keras.callbacks.ModelCheckpoint(log_folder + 'model{epoch:08d}.h5', save_freq=1) 
csv_logger = CSVLogger(log_folder + "history.csv", append=True)
logdir = log_folder + datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir)

from keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(rescale=1.0/255.0)
val_datagen = ImageDataGenerator(rescale=1.0/255.0)

train_generator = train_datagen.flow(x_train, y_train, batch_size=16, shuffle=True)
val_generator = val_datagen.flow(x_val, y_val, batch_size=2, shuffle=True)

epoch = 1000

# fit model
history = model.fit(train_generator, steps_per_epoch=len(train_generator),
    validation_data=val_generator, validation_steps=len(val_generator), callbacks=[checkpoint, csv_logger, tensorboard_callback], 
    epochs=epoch, shuffle=True, verbose=1)
model.save(log_folder + "Final_model.h5")

hist_df = pd.DataFrame(history.history) 
hist_df.to_csv(log_folder + "Final_history.csv")

# summarize history for accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig(log_folder + "accuracy_curve.jpg")

# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig(log_folder + "loss_curve.jpg")

print("Finished")
