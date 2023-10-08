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

config = tf.compat.v1.ConfigProto( device_count = {'GPU': 2 , 'CPU': 8} ) 
sess = tf.compat.v1.Session(config=config)
tf.compat.v1.keras.backend.set_session(sess)

data = np.load("data/data_16class_fixedCropError.npz")

x_train = data["x_train"]  
y_train = data["y_train"]
x_val = data["x_val"]
y_val = data["y_val"]

assert x_train.shape == (len(y_train), 224, 224, 3)
assert y_train.shape == (len(y_train), 1)
assert x_val.shape == (len(y_val), 224, 224, 3)
assert y_val.shape == (len(y_val), 1)

log_folder = "logs/efficientNetB0/32_1e-5_adam_sparseLoss_3layersAdd_TF_2/"
shutil.copy("train_keras_eff_TF_2.py", log_folder + "train_keras_eff_TF_2.py")

num_class = max(y_val)[0] + 1

# # baseModel = tf.keras.applications.EfficientNetB4(include_top=False, input_tensor=Input(shape=(224, 224, 3)), classes = 14)
# baseModel = tf.keras.applications.EfficientNetB0(weights="efficient_pretrained/efficientnetb0_notop.h5", 
#                                                  include_top=False, 
#                                                  input_tensor=Input(shape=(224, 224, 3)), classes = num_class)

# baseModel.trainable = False

# # two Dense Layer Add
# headModel = baseModel.output
# headModel = GlobalAveragePooling2D()(headModel) 
# # (7, 7) for resnet50, (5, 5) for inceptionResnetV2 # (12, 12) for 433 layer incresV2, #(7, 7) for efficientNetV2B0
# headModel = Flatten(name="flatten")(headModel)
# headModel = Dense(512, activation="relu", kernel_initializer="glorot_uniform")(headModel)
# headModel = BatchNormalization()(headModel)
# headModel_1 = Dropout(0.3)(headModel)
# headModel = Dense(256, activation="relu", kernel_initializer="glorot_uniform")(headModel_1)
# headModel = BatchNormalization()(headModel)
# headModel_2 = Dropout(0.3)(headModel)
# headModel = Dense(128, activation="relu", kernel_initializer="glorot_uniform")(headModel_2)
# headModel = BatchNormalization()(headModel)
# headModel_3 = Dropout(0.3)(headModel)
# headModel = Dense(256, activation="relu", kernel_initializer="glorot_uniform")(headModel_3)
# headModel = BatchNormalization()(headModel)
# headModel_4 = Dropout(0.3)(headModel)
# headModel = Concatenate()([headModel_1, headModel_2, headModel_3, headModel_4])
# headModel = Dense(num_class, activation="softmax")(headModel)

# model = Model(inputs=baseModel.input, outputs=headModel)

# Load model
model = load_model("logs/efficientNetB0/32_1e-3_adam_sparseLoss_3layersAdd_TF_1/best/model00000035.h5")
model.trainable = True

model.summary()
 
# optimizer = optimizers.RMSprop(lr=2e-5)
loss=keras.losses.SparseCategoricalCrossentropy()
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-5)
model.compile(loss=loss,optimizer=optimizer,metrics=['accuracy', keras.metrics.SparseCategoricalAccuracy()])
checkpoint = keras.callbacks.ModelCheckpoint(log_folder + 'models/model{epoch:08d}.h5', save_freq=1) 
csv_logger = CSVLogger(log_folder + "history.csv", append=True)
logdir = log_folder + datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir)

from keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(rescale=1.0/255.0)
val_datagen = ImageDataGenerator(rescale=1.0/255.0)

train_generator = train_datagen.flow(x_train, y_train, batch_size=32, shuffle=True)
val_generator = val_datagen.flow(x_val, y_val, batch_size=32, shuffle=True)

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
