from cgi import test
import os
import numpy as np
import cv2
from keras.models import load_model
from keras.preprocessing import image
import tensorflow as tf
import pandas as pd
import all_cams

config = tf.compat.v1.ConfigProto( device_count = {'GPU': 8 , 'CPU': 32} ) 
sess = tf.compat.v1.Session(config=config)
tf.compat.v1.keras.backend.set_session(sess)

# EfficientNet
model =load_model("logs/efficientNetB0/32_1e-5_adam_sparseLoss_3layersAdd_TF_2/models/model00000056.h5")
grad_model = tf.keras.models.Model([model.layers[0].input], [model.layers[234].output])

larvae = {0: 'anopheles_gambiae_3rd', 1: 'anopheles_gambiae_4th', 2: "anopheles_stephensi_3rd", 3: "anopheles_stephensi_4th", 
          4: 'anopheles_arabiensis_3rd', 5: 'anopheles_arabiensis_4th', 6: 'anopheles_funestus_3rd', 7: 'anopheles_funestus_4th',
          8: 'aedes_albopictus_3rd', 9: 'aedes_albopictus_4th', 10: 'culex_quinquefasciatus_3rd', 11: 'culex_quinquefasciatus_4th',
          12: 'aedes_aegypti_3rd', 13: 'aedes_aegypti_4th', 14: 'culex_tarsalis_3rd', 15: 'culex_tarsalis_4th'}

# test_df = pd.read_csv(f'data/test_14classALL.csv')

# images = []
# for i, file in enumerate(test_df['file_path']):
    
#     img = cv2.resize(cv2.imread(file), (224, 224))
#     x = image.img_to_array(img) / 255
#     images.append(x)

# res = model.predict(np.array(images).astype('float32')).tolist()
# predicted = [larvae[r.index(max(r))] for r in res]
# corrected = int(sum(np.array([list(test_df["class_name"])[i] == predicted[i] for i in range(len(predicted))])  * 1))
# incorrected = int(len(predicted) - sum(np.array([list(test_df["class_name"])[i] == predicted[i] for i in range(len(predicted))])  * 1))

for i, file in enumerate(test_df['file_path']):
    
    img = cv2.resize(cv2.imread(file), (224, 224))
    x = image.img_to_array(img) / 255
    res = model.predict(np.array([x]).astype('float32')).tolist()[0]
    predicted = larvae[res.index(max(res))]
    percentage = str(max(res) * 100)
    
    pred = predicted == test_df["class_name"][i]
    if pred: pred_correct = "correct" 
    else: pred_correct = "incorrect"
    
    name = test_df['file_name'][i].split(".jpg")[0]
    name = name + "_predicted_" + predicted + f"_{str(pred)}.jpg"

    print(name)

    img = cv2.imread(file)
    height, width, _ = img.shape

    threshold = 0
    
    ## grad- cam
    superimposed_img = all_cams.grad_cam_keras(model, x, height, width, threshold = threshold)
    alpha = 0.5
    superimposed_img = superimposed_img * alpha + img
    cv2.imwrite(f"test_augmented_cam/test/cam/{pred_correct}/{name}", superimposed_img)
    
    ## cam
    superimposed_img = all_cams.cam(model, grad_model, x, "efficientNet", height, width, threshold = threshold)
    alpha = 0.5
    superimposed_img = superimposed_img * alpha + img
    cv2.imwrite(f"test_augmented_cam/test/grad_cam/{pred_correct}/{name}", superimposed_img)


for image_name in os.listdir("random_testImages/"):

    img = cv2.resize(cv2.imread("random_testImages/" + image_name), (224, 224))
    x = image.img_to_array(img) / 255
    res = model.predict(np.array([x]).astype('float32')).tolist()[0]
    predicted = larvae[res.index(max(res))]
    percentage = str(max(res) * 100)

    print(image_name + " is " + predicted + " with a probability of " + percentage + "%") 
    
    name = image_name.split(".")[0]
    name = name + "_predicted_" + predicted + f".jpg"
    
    img = cv2.imread("random_testImages/" + image_name)
    height, width, _ = img.shape

    threshold = 0
    
    ## grad- cam
    superimposed_img = all_cams.grad_cam_keras(model, x, height, width, threshold = threshold)
    alpha = 0.5
    superimposed_img = superimposed_img * alpha + img
    cv2.imwrite(f"random_testImages/cam/{name}", superimposed_img)
    
    ## cam
    superimposed_img = all_cams.cam(model, grad_model, x, "efficientNet", height, width, threshold = threshold)
    alpha = 0.5
    superimposed_img = superimposed_img * alpha + img
    cv2.imwrite(f"random_testImages/grad_cam/{name}", superimposed_img)
    
    


# for i, file in enumerate(val_df['file_path']):
#     img = cv2.resize(cv2.imread(file), (224, 224))
#     x = image.img_to_array(img) / 255
#     res = model.predict(np.array([x]).astype('float32')).tolist()[0]
#     predicted = larvae[res.index(max(res))]
#     percentage = str(max(res) * 100)
#     pred = predicted == val_df["class_name"][i]
#     name = val_df["class_name"][i] + f"_{i+1}_{str(pred)}.jpg"

#     print(name + " is " + predicted + " with a probability of " + str(percentage) + "%")

#     last_conv_layer_output = grad_model(np.array([x]).astype('float32'))
#     last_conv_layer_output = last_conv_layer_output[0]

#     class_weights = model.layers[235].get_weights()[0]
#     heatmap = tf.reduce_mean(class_weights * last_conv_layer_output, axis=(2))

#     heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
#     heatmap = np.array(heatmap)
#     heatmap = np.uint8(255 * heatmap)

#     heatmap_ = heatmap

#     img = cv2.imread(file)
#     height, width, _ = img.shape
#     heatmap_ = cv2.applyColorMap(cv2.resize(heatmap_,(width, height)), cv2.COLORMAP_JET)

#     alpha = 0.5
#     superimposed_img = heatmap_ * alpha + img

#     if height > 1000:
#         cv2.putText(superimposed_img, f'{predicted} ({str(percentage)}%)', (20, 140), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 0.9 * int(height/500), (255,255,255), 3, cv2.LINE_AA)
#     else:
#         cv2.putText(superimposed_img, f'{predicted} ({str(percentage)}%)', (20, 20), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 0.9, (255,165,0), 1, cv2.LINE_AA)

#     cv2.imwrite(f"output/val/{name}", superimposed_img)


# for i, file in enumerate(test_df['file_path']):
    
#     img = cv2.resize(cv2.imread(file), (224, 224))
#     x = image.img_to_array(img) / 255
#     res = model.predict(np.array([x]).astype('float32')).tolist()[0]
#     predicted = larvae[res.index(max(res))]
#     percentage = str(max(res) * 100)
#     pred = predicted == test_df["class_name"][i]
#     name = test_df["class_name"][i] + f"_{i+1}_{str(pred)}.jpg"

#     print(name + " is " + predicted + " with a probability of " + str(percentage) + "%")

#     last_conv_layer_output = grad_model(np.array([x]).astype('float32'))
#     last_conv_layer_output = last_conv_layer_output[0]

#     class_weights = model.layers[235].get_weights()[0]
#     heatmap = tf.reduce_mean(class_weights * last_conv_layer_output, axis=(2))

#     heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
#     heatmap = np.array(heatmap)
#     heatmap = np.uint8(255 * heatmap)

#     heatmap_ = heatmap

#     img = cv2.imread(file)
#     height, width, _ = img.shape
#     heatmap_ = cv2.applyColorMap(cv2.resize(heatmap_,(width, height)), cv2.COLORMAP_JET)

#     alpha = 0.5
#     superimposed_img = heatmap_ * alpha + img

#     if height > 1000:
#         cv2.putText(superimposed_img, f'{predicted} ({str(percentage)}%)', (20, 140), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 0.9 * int(height/500), (255,255,255), 3, cv2.LINE_AA)
#     else:
#         cv2.putText(superimposed_img, f'{predicted} ({str(percentage)}%)', (20, 20), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 0.9, (255,165,0), 1, cv2.LINE_AA)

#     cv2.imwrite(f"output/test/{name}", superimposed_img)

# x = []
# for i, file in enumerate(val_df['file_path']):
#     img = cv2.resize(cv2.imread(file), (224, 224))
#     x.append(image.img_to_array(img) / 255)
# res = model.predict(np.array(x).astype('float32')).tolist()
# predicted = [larvae[r.index(max(r))] for r in res]

# wrong_prediction = []
# for i, file in enumerate(val_df['file_path']):
    
#     if predicted[i] != val_df["class_name"][i]:  
        
#         wrong_prediction.append(val_df['file_name'][i])
#         name = val_df['file_name'][i][:-4] + f"_predicted_{str(predicted[i])}.jpg"

#         print(name + " is predicted as " + predicted[i]) #  + " with a probability of " + str(percentage) + "%"
        
#         img = cv2.imread(file)
#         height, width, _ = img.shape

#         heatmap = all_cams.grad_cam_mine(grad_model, x[i], height, width)
#         alpha = 0.5
#         superimposed_img = heatmap * alpha + img
#         cv2.imwrite(f"wrong_prediction/val/cam_{name}", superimposed_img)


# x = []
# for i, file in enumerate(test_df['file_path']):
#     img = cv2.resize(cv2.imread(file), (224, 224))
#     x.append(image.img_to_array(img) / 255)
# res = model.predict(np.array(x).astype('float32')).tolist()
# predicted = [larvae[r.index(max(r))] for r in res]

# wrong_prediction = []
# for i, file in enumerate(test_df['file_path']):
    
#     if predicted[i] != test_df["class_name"][i]:  
        
#         wrong_prediction.append(test_df['file_name'][i])
#         name = test_df['file_name'][i][:-4] + f"_predicted_{str(predicted[i])}.jpg"

#         print(test_df['file_name'][i] + " is predicted as " + predicted[i]) #  + " with a probability of " + str(percentage) + "%"
        
#         img = cv2.imread(file)
#         # height, width, _ = img.shape
        
#         # heatmap = all_cams.grad_cam_mine(grad_model, x[i], height, width)
#         # alpha = 0.5
#         # superimposed_img = heatmap * alpha + img
#         cv2.imwrite(f"wrong_prediction/test/{name}", img)

print("Finished")

