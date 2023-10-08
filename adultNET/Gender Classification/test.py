from keras.models import load_model
import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix
import os, shutil

# config = tf.compat.v1.ConfigProto( device_count = {'GPU': 2 , 'CPU': 16} ) 
# sess = tf.compat.v1.Session(config=config)
# tf.compat.v1.keras.backend.set_session(sess)

log_folder = "logs/efficientNetB0/16_1e-5_adam_sparseLoss_2layers_1/models"
# shutil.copy("test.py", log_folder + "test.py")

adult = {'female': 0, 'male': 1}

# import CSVLogger as log
# logger = log.CSVLogger(os.path.join(log_folder,'testing_438to.csv')) 

data = np.load("data/data.npz")

x_val = data["x_val"]
y_val = data["y_val"]
assert x_val.shape == (36, 224, 224, 3)
assert y_val.shape == (36, 1)
x_val = x_val / 255.0

x_test = data["x_test"]
y_test = data["y_test"]
assert x_test.shape == (36, 224, 224, 3)
assert y_test.shape == (36, 1)
x_test = x_test / 255.0

# logger.on_train_begin()
files = os.listdir(log_folder)
files.sort()

for e, m in enumerate(files):
    if(".h5" in m):
        if(int(m[5:][:-3]) == 302):
                model = load_model(f'{log_folder}/{m}')
                res = model.predict(x_test)
                res = np.argmax(res, axis=1).reshape((36, 1))
                cm_ = confusion_matrix(y_test,res)
                accuracy_ = (sum(res == y_test) / len(y_test))[0] * 100

                res = model.predict(x_val)
                res = np.argmax(res, axis=1).reshape((36, 1))
                cm = confusion_matrix(y_val,res)
                accuracy = (sum(res == y_val) / len(y_val))[0] * 100

                # print(f"{m} >> \n\n{cm_} \n\n Accuracy - {accuracy, accuracy_}%\n")
                
                logs = {'epoch' : e+1,
                        'model' : m,
                        'val_cm' : cm,
                        'val_accuracy' : accuracy,
                        'test_cm' : cm_,
                        'test_accuracy' : accuracy_}
                print(logs)
#                 logger.on_epoch_end(e, logs)

# logger.on_train_end()
print("Shesh")

# cm = confusion_matrix(np.argmax(y_test, axis=1), y_pred)
# print(cm)
# target = ["Category {}".format(i) for i in range(n_classes)]
# print(classification_report(np.argmax(y_test, axis=1), y_pred, target_names=target))

