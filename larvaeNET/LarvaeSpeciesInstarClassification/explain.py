from cgi import test
import os
import numpy as np
import cv2
from keras.models import load_model
from keras.preprocessing import image
import tensorflow as tf
import all_cams
from tensorflow.keras import layers


# config = tf.compat.v1.ConfigProto( device_count = {'GPU': 8 , 'CPU': 32} )
# sess = tf.compat.v1.Session(config=config)
# tf.compat.v1.keras.backend.set_session(sess)

# EfficientNet
model =load_model("logs/efficientNetB0/32_1e-5_adam_sparseLoss_3layersAdd_1/model00000151.h5")

# last_conv_layer = next(x for x in model.layers[::-1] if isinstance(x, layers.Conv2D))
# target_layer = model.get_layer(last_conv_layer.name)
# grad_model = tf.keras.models.Model([model.layers[0].input], [target_layer.output])

grad_model = tf.keras.models.Model([model.layers[0].input], [model.layers[471].output])

larvae = {0: 'anopheles_gambiae_3rd', 1: 'anopheles_gambiae_4th', 2: "anopheles_stephensi_3rd", 3: "anopheles_stephensi_4th", 
          4: 'anopheles_arabiensis_3rd', 5: 'anopheles_arabiensis_4th', 6: 'anopheles_funestus_3rd', 7: 'anopheles_funestus_4th',
          8: 'aedes_albopictus_3rd', 9: 'aedes_albopictus_4th', 'culex_quinquefasciatus_3rd': 10, 'culex_quinquefasciatus_4th': 11,
          12: 'aedes_aegypti_3rd', 13: 'aedes_aegypti_4th'}

file = "original.jpeg"
img = cv2.resize(cv2.imread(file), (224, 224))
x = image.img_to_array(img) / 255
res = model.predict(np.array([x]).astype('float32')).tolist()[0]
predicted = larvae[res.index(max(res))]
percentage = str(max(res) * 100)

print("This is predicted as " + predicted + " with a probability of " + str(percentage) + "%")

img = cv2.imread(file)
height, width, _ = img.shape

threshold = 0

# ## cam
# superimposed_img = all_cams.cam(model, grad_model, x, "efficientNet", height, width, threshold = threshold)

## grad- cam
superimposed_img = all_cams.grad_cam_keras(model, x, height, width, threshold = threshold)

alpha = 0.5
superimposed_img = superimposed_img * alpha + img

# if height > 1000:
#     cv2.putText(superimposed_img, f'{predicted} ({str(percentage)}%)', (20, 140), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 0.9 * int(height/500), (255,255,255), 3, cv2.LINE_AA)
# else:
#     cv2.putText(superimposed_img, f'{predicted} ({str(percentage)}%)', (20, 20), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 0.9, (255,165,0), 1, cv2.LINE_AA)

## cam
for threshold in range(110, 255, 5):
    superimposed_img = all_cams.cam(model, grad_model, x, "efficientNet", height, width, threshold = threshold)

    alpha = 0.5
    superimposed_img = superimposed_img * alpha + img
    cv2.imwrite(f"test_augmented_cam/cam/{threshold}_cam_{file}", superimposed_img)

## grad- cam
for threshold in range(110, 255, 5):
    superimposed_img = all_cams.grad_cam_keras(model, x, height, width, threshold = threshold)

    alpha = 0.5
    superimposed_img = superimposed_img * alpha + img
    cv2.imwrite(f"test_augmented_cam/grad_cam_keras/{threshold}_gradCam_{file}", superimposed_img)

print("Finished")
