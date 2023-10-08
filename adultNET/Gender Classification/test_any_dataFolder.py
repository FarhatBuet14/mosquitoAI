from cgi import test
import os
import numpy as np
import cv2
from keras.models import load_model
from keras.preprocessing import image
import tensorflow as tf
import pandas as pd

# EfficientNet
model =load_model("logs/efficientNetB0/16_1e-5_adam_sparseLoss_2layers_1/models/model00000302.h5")
grad_model = tf.keras.models.Model([model.layers[0].input], [model.layers[234].output])

larvae = {0: 'female', 1: 'male'}

val_df = pd.read_csv(f'data/val.csv')
test_df = pd.read_csv(f'data/test.csv')

for i, file in enumerate(val_df['file_path']):
    img = cv2.resize(cv2.imread(file), (224, 224))
    x = image.img_to_array(img) / 255
    res = model.predict(np.array([x]).astype('float32')).tolist()[0]
    predicted = larvae[res.index(max(res))]
    percentage = str(max(res) * 100)
    pred = predicted == val_df["class_name"][i]
    name = val_df["class_name"][i] + f"_{i+1}_{str(pred)}.jpg"

    print(name + " is " + predicted + " with a probability of " + str(percentage) + "%")

    # last_conv_layer_output = grad_model(np.array([x]).astype('float32'))
    # last_conv_layer_output = last_conv_layer_output[0]

    # class_weights = model.layers[235].get_weights()[0]
    # heatmap = tf.reduce_mean(class_weights * last_conv_layer_output, axis=(2))

    # heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    # heatmap = np.array(heatmap)
    # heatmap = np.uint8(255 * heatmap)

    last_conv_layer_output = grad_model(np.array([x]).astype('float32'))
    pooled_grads = tf.reduce_mean(last_conv_layer_output, axis=(0, 1, 2))
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    heatmap = np.array(heatmap)
    heatmap = np.uint8(255 * heatmap)

    heatmap_ = heatmap * (heatmap > 110)

    img = cv2.imread(file)
    height, width, _ = img.shape
    heatmap_ = cv2.applyColorMap(cv2.resize(heatmap_,(width, height)), cv2.COLORMAP_JET)

    alpha = 0.5
    superimposed_img = heatmap_ * alpha + img

    if height > 1000:
        cv2.putText(superimposed_img, f'{predicted} ({str(percentage)}%)', (20, 140), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 0.9 * int(height/500), (255,255,255), 3, cv2.LINE_AA)
    else:
        cv2.putText(superimposed_img, f'{predicted} ({str(percentage)}%)', (20, 20), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 0.9, (255,165,0), 1, cv2.LINE_AA)

    cv2.imwrite(f"output/val/{name}", superimposed_img)


for i, file in enumerate(test_df['file_path']):
    
    img = cv2.resize(cv2.imread(file), (224, 224))
    x = image.img_to_array(img) / 255
    res = model.predict(np.array([x]).astype('float32')).tolist()[0]
    predicted = larvae[res.index(max(res))]
    percentage = str(max(res) * 100)
    pred = predicted == test_df["class_name"][i]
    name = test_df["class_name"][i] + f"_{i+1}_{str(pred)}.jpg"

    print(name + " is " + predicted + " with a probability of " + str(percentage) + "%")

    # last_conv_layer_output = grad_model(np.array([x]).astype('float32'))
    # last_conv_layer_output = last_conv_layer_output[0]

    # class_weights = model.layers[235].get_weights()[0]
    # heatmap = tf.reduce_mean(class_weights * last_conv_layer_output, axis=(2))

    # heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    # heatmap = np.array(heatmap)
    # heatmap = np.uint8(255 * heatmap)

    last_conv_layer_output = grad_model(np.array([x]).astype('float32'))
    pooled_grads = tf.reduce_mean(last_conv_layer_output, axis=(0, 1, 2))
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    heatmap = np.array(heatmap)
    heatmap = np.uint8(255 * heatmap)

    heatmap_ = heatmap * (heatmap > 110)

    img = cv2.imread(file)
    height, width, _ = img.shape
    heatmap_ = cv2.applyColorMap(cv2.resize(heatmap_,(width, height)), cv2.COLORMAP_JET)

    alpha = 0.5
    superimposed_img = heatmap_ * alpha + img

    if height > 1000:
        cv2.putText(superimposed_img, f'{predicted} ({str(percentage)}%)', (20, 140), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 0.9 * int(height/500), (255,255,255), 3, cv2.LINE_AA)
    else:
        cv2.putText(superimposed_img, f'{predicted} ({str(percentage)}%)', (20, 20), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 0.9, (255,165,0), 1, cv2.LINE_AA)

    cv2.imwrite(f"output/test/{name}", superimposed_img)
