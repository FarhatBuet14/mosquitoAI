from keras.models import load_model
from keras.preprocessing import image
import numpy as np
import tensorflow as tf
import cv2, os
import all_cams

model =load_model("logs/efficientNetB0/32_1e-5_adam_sparseLoss_3layersAdd_TF_2/models/model00000262.h5")
grad_model = tf.keras.models.Model([model.layers[0].input], [model.layers[234].output])

larvae = {0: 'not_anopheles', 1: 'anopheles'}

directory = "test_augmented/"

results = []
for image_name in os.listdir(directory):

    img = cv2.resize(cv2.imread(directory + image_name), (224, 224))
    x = image.img_to_array(img) / 255
    res = model.predict(np.array([x]).astype('float32')).tolist()[0]
    results.append(res)
    predicted = larvae[res.index(max(res))]
    percentage = str(max(res) * 100)

    print(image_name + " is " + predicted + " with a probability of " + percentage + "%") 
    
    img = cv2.imread(directory + image_name)
    height, width, _ = img.shape
    threshold = 0
    alpha = 0.5

    ## cam
    superimposed_img = all_cams.cam(model, grad_model, x, "efficientNet", height, width, threshold = threshold)
    superimposed_img = superimposed_img * alpha + img
    cv2.imwrite(f"results/madagascar/cam/{threshold}_cam_{image_name}", superimposed_img)

    ## grad- cam
    superimposed_img = all_cams.grad_cam_keras(model, x, height, width, threshold = threshold)
    superimposed_img = superimposed_img * alpha + img
    cv2.imwrite(f"results/madagascar/grad_cam/{threshold}_gradCam_{image_name}", superimposed_img)
    
    ## grad- cam- mine
    superimposed_img = all_cams.grad_cam_mine(grad_model, x, height, width, threshold = threshold)
    superimposed_img = superimposed_img * alpha + img
    cv2.imwrite(f"results/madagascar/grad_cam_mine/{threshold}_gradCam_{image_name}", superimposed_img)

results = np.array(results)
mean = results.mean(axis = 0)

predicted = larvae[np.argmax(mean)]
percentage = str(max(mean) * 100)
print("This image is " + predicted + " with a average probability of " + percentage + "%") 


directory = "madagascar_nonAnopheles/"

for image_name in os.listdir(directory):

    img = cv2.resize(cv2.imread(directory + image_name), (224, 224))
    x = image.img_to_array(img) / 255
    res = model.predict(np.array([x]).astype('float32')).tolist()[0]
    predicted = larvae[res.index(max(res))]
    percentage = str(max(res) * 100)

    print(image_name + " is " + predicted + " with a probability of " + percentage + "%") 
    
    img = cv2.imread(directory + image_name)
    height, width, _ = img.shape
    threshold = 0
    alpha = 0.5

    ## cam
    superimposed_img = all_cams.cam(model, grad_model, x, "efficientNet", height, width, threshold = threshold)
    superimposed_img = superimposed_img * alpha + img
    cv2.imwrite(f"results/madagascar_nonAnopheles/cam/{threshold}_cam_{image_name}", superimposed_img)

    ## grad- cam
    superimposed_img = all_cams.grad_cam_keras(model, x, height, width, threshold = threshold)
    superimposed_img = superimposed_img * alpha + img
    cv2.imwrite(f"results/madagascar_nonAnopheles/grad_cam/{threshold}_gradCam_{image_name}", superimposed_img)
    
    ## grad- cam- mine
    superimposed_img = all_cams.grad_cam_mine(grad_model, x, height, width, threshold = threshold)
    superimposed_img = superimposed_img * alpha + img
    cv2.imwrite(f"results/madagascar_nonAnopheles/grad_cam_mine/{threshold}_gradCam_{image_name}", superimposed_img)


print("Finished")