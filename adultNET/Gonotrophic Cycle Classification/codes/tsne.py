from sklearn.manifold import TSNE
import numpy as np
np.random.seed(9)
import matplotlib.pyplot as plt
import tensorflow as tf
import seaborn as sns
import pandas as pd
from keras.models import load_model
import argparse
import keras

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

if param.name == "ConvNeXtTiny":
    import keras.applications.convnext as cvx
    model = tf.keras.models.load_model(param.model_directory, compile = False, custom_objects={'LayerScale':cvx.LayerScale})
else:
    model = tf.keras.models.load_model(param.model_directory, compile = False)

optimizer = tf.keras.optimizers.Adam(learning_rate = 0.00001)
loss=keras.losses.SparseCategoricalCrossentropy()
model.compile(loss=loss,optimizer=optimizer,metrics=['accuracy', keras.metrics.SparseCategoricalAccuracy()])

res = model.predict(x_test)
y_pred = np.argmax(res, axis=1).reshape((len(x_test), 1))
y_pred = list(y_pred.reshape(-1, ))
matched = list([int(y_pred[i] == y_test[i]) for i, elem in enumerate(y_pred)])

markers = {1: "correct", 0: "incorrect"}

last_conv_layer = {"EfficientNetB0": 234, "ConvNeXtTiny": 142, "ResNet50": 171, "MobileNetV2": 151}

grad_model = tf.keras.models.Model([model.layers[0].input], [model.layers[last_conv_layer[param.model_name]].output])
total_features = grad_model.predict(x_test)
features = total_features.reshape(total_features.shape[0], total_features.shape[1]*total_features.shape[2]*total_features.shape[3])

labels_adult = {0: 'unfed', 1: 'gravid', 2: "semi-gravid", 3: "fully fed"}

tsne = TSNE(n_components=2, verbose=0, perplexity=26, n_iter=300)
output = tsne.fit_transform(features, y_test)

tsne_result_df = pd.DataFrame({'tsne_1': output[:,0], 'tsne_2': output[:,1], 'stages': [labels_adult[elem] for elem in list(y_test.reshape(-1, ))],
                                'predicted': [markers[elem] for elem in matched]})

fig, ax = plt.subplots(figsize = (8, 5))
sns.scatterplot(x='tsne_1', y='tsne_2', hue='stages', style='predicted', data=tsne_result_df, ax=ax)
lim = (output.min()-5, output.max()+5)
ax.set_xlim(lim)
ax.set_ylim(lim)
ax.set_aspect('equal')
ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0)

plt.savefig(f'tsne.png')

print("Finished")
