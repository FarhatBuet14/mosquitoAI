from keras.models import load_model
import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix
import os, shutil

config = tf.compat.v1.ConfigProto( device_count = {'GPU': 2 , 'CPU': 16} ) 
sess = tf.compat.v1.Session(config=config)
tf.compat.v1.keras.backend.set_session(sess)

log_folder = "logs/efficientNetB0/8_1e-5_adam_sparseLoss_3layersAdd_TF_2/"
shutil.copy("test.py", os.path.join(log_folder , "test.py"))

larvae = {0: 'not_anopheles', 1: 'anopheles'}

import CSVLogger as log

data = np.load("data/data_anophelesClassifier_16class.npz")

x_val = data["x_val"]
y_val = data["y_val"]
assert x_val.shape == (len(y_val), 224, 224, 3)
assert y_val.shape == (len(y_val), 1)
x_val = x_val / 255.0
 
x_test = data["x_test"]
y_test = data["y_test"]
assert x_test.shape == (len(y_test), 224, 224, 3)
assert y_test.shape == (len(y_test), 1)
x_test = x_test / 255.0

# logger.on_train_begin()
files = os.listdir(os.path.join(log_folder, "models"))
files.sort()

for e, m in enumerate(files):
    if(".h5" in m):
        if(int(m[5:][:-3]) == 216):
                model = load_model(f'{log_folder}/models/{m}')
                res = model.predict(x_test)
                res = np.argmax(res, axis=1).reshape((len(y_test), 1))
                cm_ = confusion_matrix(y_test,res)
                accuracy_ = (sum(res == y_test) / len(y_test))[0] * 100

                res = model.predict(x_val)
                res = np.argmax(res, axis=1).reshape((len(y_val), 1))
                cm = confusion_matrix(y_val,res)
                accuracy = (sum(res == y_val) / len(y_val))[0] * 100

                print(f"{m} >> \n\n{cm_} \n\n Accuracy - {accuracy, accuracy_}%\n")

                logs = {'epoch' : e+1,
                        'model' : m,
                        'val_cm' : cm,
                        'val_accuracy' : accuracy,
                        'test_cm' : cm_,
                        'test_accuracy' : accuracy_}
                print(logs)
                break

print("Shesh")
