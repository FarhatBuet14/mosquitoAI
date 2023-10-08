from tsnecuda import TSNE
import numpy as np
np.random.seed(9)
import matplotlib.pyplot as plt

from keras.models import load_model
import numpy as np
import tensorflow as tf
import pandas as pd

larvae = {0: 'anopheles_gambiae_3rd', 1: 'anopheles_gambiae_4th', 2: "anopheles_stephensi_3rd", 3: "anopheles_stephensi_4th", 
          4: 'anopheles_arabiensis_3rd', 5: 'anopheles_arabiensis_4th', 6: 'anopheles_funestus_3rd', 7: 'anopheles_funestus_4th',
          8: 'aedes_albopictus_3rd', 9: 'aedes_albopictus_4th', 10: 'culex_quinquefasciatus_3rd', 11: 'culex_quinquefasciatus_4th',
          12: 'aedes_aegypti_3rd', 13: 'aedes_aegypti_4th'}

model = load_model("model00000151.h5")
grad_model = tf.keras.models.Model([model.layers[0].input], [model.layers[475].output])

data = np.load("data/data_14class.npz")
x_test = data["x_test"]
y_test = data["y_test"]
assert x_test.shape == (168, 224, 224, 3)
assert y_test.shape == (168, 1)
x_test = x_test / 255.0

total_features = grad_model.predict(x_test)
cls_pred = model.predict(x_test)

colors_per_class = ["Blue", "cornflowerblue", "Red", "lightcoral", "cyan", "teal", \
                    "green", "palegreen", "black", "grey", "peru", "darkorange", \ 
                    "fuchsia", "purple"]

n_components = np.arange(50)
perplexity = np.arange(50)

for comp in n_components:
    for per in perplexity:
        
        tsne = TSNE(n_components=comp, perplexity=per, learning_rate=10).fit_transform(total_features)

        def scale_to_01_range(x):
            value_range = (np.max(x) - np.min(x))
            starts_from_zero = x - np.min(x)
            return starts_from_zero / value_range

        tx = tsne[:, 0]
        ty = tsne[:, 1]
        tx = scale_to_01_range(tx)
        ty = scale_to_01_range(ty)

        #---- Class wise TSNE
        fig = plt.figure(figsize=(12, 6.33))
        ax = fig.add_subplot(111)

        for label, color in enumerate(colors_per_class):
            indices = [i for i, l in enumerate(y_test) if l == label]
            current_tx = np.take(tx, indices)
            current_ty = np.take(ty, indices)
            ax.scatter(current_tx, current_ty, c=color)

        ax.legend(loc='best')
        plt.savefig(f'tsne/tsne_com{comp}_per{per}.png')



from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt 

# first reduce dimensionality before feeding to t-sne
pca = PCA(n_components=50)
X_pca = pca.fit_transform(total_features) 
# randomly sample data to run quickly
rows = np.arange(70000)
np.random.shuffle(rows)
n_select = 10000 
# reduce dimensionality with t-sne
tsne = TSNE(n_components=2, verbose=1, perplexity=50, n_iter=1000, learning_rate=200)
tsne_results = tsne.fit_transform(X_pca[rows[:n_select],:])
# visualize
df_tsne = pd.DataFrame(tsne_results, columns=['comp1', 'comp2'])
df_tsne['label'] = y[rows[:n_select]]
sns.lmplot(x='comp1', y='comp2', data=df_tsne, hue='label', fit_reg=False)
plt.savefig(f'bla.png')
