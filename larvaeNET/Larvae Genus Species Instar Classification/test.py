from keras.models import load_model
import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix
import os, shutil

config = tf.compat.v1.ConfigProto( device_count = {'GPU': 2 , 'CPU': 16} ) 
sess = tf.compat.v1.Session(config=config)
tf.compat.v1.keras.backend.set_session(sess)

log_folder = "logs/efficientNetB0/32_1e-5_adam_sparseLoss_3layersAdd_1"
shutil.copy("test.py", os.path.join(log_folder , "test.py"))

larvae = {0: 'anopheles_gambiae_3rd', 1: 'anopheles_gambiae_4th', 2: "anopheles_stephensi_3rd", 3: "anopheles_stephensi_4th", 
          4: 'anopheles_arabiensis_3rd', 5: 'anopheles_arabiensis_4th', 6: 'anopheles_funestus_3rd', 7: 'anopheles_funestus_4th',
          8: 'aedes_albopictus_3rd', 9: 'aedes_albopictus_4th', 10: 'culex_quinquefasciatus_3rd', 11: 'culex_quinquefasciatus_4th',
          12: 'aedes_aegypti_3rd', 13: 'aedes_aegypti_4th'}

import CSVLogger as log
# logger = log.CSVLogger(os.path.join(log_folder,'testing_85to.csv')) 

data = np.load("data/data_14class.npz")

x_val = data["x_val"]
y_val = data["y_val"]
assert x_val.shape == (126, 224, 224, 3)
assert y_val.shape == (126, 1)
x_val = x_val / 255.0
 
x_test = data["x_test"]
y_test = data["y_test"]
assert x_test.shape == (168, 224, 224, 3)
assert y_test.shape == (168, 1)
x_test = x_test / 255.0

# logger.on_train_begin()
files = os.listdir(os.path.join(log_folder))
files.sort()

for e, m in enumerate(files):
    if(".h5" in m):
        if(int(m[5:][:-3]) == 151):
                model = load_model(f'{log_folder}/{m}')
                res = model.predict(x_test)
                res = np.argmax(res, axis=1).reshape((168, 1))
                cm_ = confusion_matrix(y_test,res)
                accuracy_ = (sum(res == y_test) / len(y_test))[0] * 100

                res = model.predict(x_val)
                res = np.argmax(res, axis=1).reshape((126, 1))
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
                # logger.on_epoch_end(e, logs)

# logger.on_train_end()
print("Shesh")

# cm = confusion_matrix(np.argmax(y_test, axis=1), y_pred)
# print(cm)
# target = ["Category {}".format(i) for i in range(n_classes)]
# print(classification_report(np.argmax(y_test, axis=1), y_pred, target_names=target))
