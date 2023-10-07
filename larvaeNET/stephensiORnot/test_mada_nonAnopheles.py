from keras.models import load_model
import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix
import os, shutil
import pandas as pd

from cgi import test
import os
import numpy as np
import cv2
from keras.models import load_model
from keras.preprocessing import image
import tensorflow as tf
import all_cams
from tensorflow.keras import layers
 
larvae = {0: 'not_anopheles', 1: 'anopheles'}

model = load_model(f'logs/efficientNetB0/8_1e-5_adam_sparseLoss_3layersAdd_TF_2/models/model00000062.h5')
# grad_model = tf.keras.models.Model([model.layers[0].input], [model.layers[234].output])


dataset = os.listdir("madagascar_nonAnopheles_3perSpec")
if ".DS_Store" in dataset: dataset.remove(".DS_Store")

images = []
for image_name in dataset:

    img = cv2.resize(cv2.imread("madagascar_nonAnopheles_3perSpec/" + image_name), (224, 224))
    images.append(image.img_to_array(img) / 255)

result = model.predict(np.array(images).astype('float32')).tolist()
predicted = [larvae[res.index(max(res))] for res in result]

truth = []
for nam in dataset:
    if "ano" in nam: truth.append("anopheles")
    else: truth.append("not_anopheles")
    

names = dataset

count = 0
matched = []
for i, res in enumerate(predicted):
    if(truth[i] == predicted[i]): 
        count += 1
        matched.append(True)
    else: matched.append(False)
print(count/len(truth))

df = pd.DataFrame(list(zip(names, truth, predicted, matched)), \
    columns = ["name", "truth", "predicted", "matched_binary"])
df.to_csv(os.path.join('madagascar_nonAnopheles_3perSpec.csv'))


cm = confusion_matrix(truth, predicted, labels = list(larvae.values()))
print(cm)


wrong = list(df.loc[df["matched"] == False]["matched_binary"])

gam = 0
ara = 0
albo = 0
quinq = 0

for name in wrong:
    if("gam" in name): gam += 1
    elif("ara" in name): ara += 1
    elif("albo" in name): albo += 1
    elif("quinq" in name): quinq += 1

data = pd.read_csv("classification_data_mada_all.csv")
updated_data = data.loc[data['name'].isin(names)]
predicted = []
for name in updated_data["name"]:
    predicted.append(list(df.loc[df["name"] == name]["predicted"])[0])

updated_data["predicted"] = predicted
updated_data.to_csv(os.path.join('madagascar_nonAnopheles_3perSpec_allInfo.csv'))


### ---- 8class model

ano_model = load_model(f'8class_914.h5')

larvae8 = {0: 'anopheles_gambiae_3rd', 1: 'anopheles_gambiae_4th', 2: "anopheles_stephensi_3rd", 3: "anopheles_stephensi_4th", 
          4: 'anopheles_arabiensis_3rd', 5: 'anopheles_arabiensis_4th', 6: 'anopheles_funestus_3rd', 7: 'anopheles_funestus_4th'}

ano_result = ano_model.predict(np.array(images).astype('float32')).tolist()
ano_predicted = [larvae8[res.index(max(res))] for res in ano_result]

overall_truth = []
for nam in dataset:
    if("ano" not in nam): overall_truth.append("not_anopheles")
    else: overall_truth.append("_".join(nam.split("_")[:3]))

overall_prediction = []
for i in range(len(overall_truth)):
    if predicted[i] == 'not_anopheles': overall_prediction.append(predicted[i])
    else: overall_prediction.append(ano_predicted[i])

truth_all = ["_".join(nam.split("_")[:3]) for nam in dataset]

count = 0
overall_matched = []
for i in range(len(overall_prediction)):
    if(overall_truth[i] == overall_prediction[i]): 
        count += 1
        overall_matched.append(True)
    else: overall_matched.append(False)
print(count/len(overall_truth))

df = pd.DataFrame(list(zip(names, truth, predicted, matched, overall_truth, overall_prediction, overall_matched)), \
    columns = ["name", "truth", "predicted", "matched_binary", "overall_truth", "overall_prediction", "overall_matched"])

labels = ['anopheles_gambiae_3rd', 'anopheles_gambiae_4th', "anopheles_stephensi_3rd", "anopheles_stephensi_4th", 
           'anopheles_arabiensis_3rd', 'anopheles_arabiensis_4th', 'anopheles_funestus_3rd', 'anopheles_funestus_4th', "not_anopheles"]
cm = confusion_matrix(overall_truth, overall_prediction, labels = labels)
print(cm)

gam = 0
ara = 0
albo = 0
quinq = 0

for i, pred in enumerate(overall_prediction):
    if pred == "fun":
        if("gam" in truth_all[i]): gam += 1
        elif("ara" in truth_all[i]): ara += 1
        elif("albo" in truth_all[i]): albo += 1
        elif("quinq" in truth_all[i]): quinq += 1
            
data = pd.read_csv("classification_data_mada_all.csv")
updated_data = data.loc[data['name'].isin(names)]
bla = []
ano_bla = []
for name in updated_data["name"]:
    bla.append(list(df.loc[df["name"] == name]["predicted"])[0])
    ano_bla.append(list(df.loc[df["name"] == name]["overall_prediction"])[0])

updated_data["predicted_binary"] = bla
updated_data["predicted_overall"] = ano_bla

updated_data.to_csv(os.path.join('madagascar_nonAnopheles_3perSpec_allInfo.csv'))

# csv_data = pd.read_csv("madagascar_nonAnopheles.csv")
# truth = list(csv_data["truth"])
# predicted = list(csv_data["predicted"])
# names = list(csv_data["name"])
# matched = list(csv_data["matched"])
# matched_genus = list(csv_data["matched_genus"])
# matched_genusSpecies = list(csv_data["matched_genusSpecies"])


# only_matched = csv_data[csv_data["matched"]]

# truth = list(only_matched["truth"])
# predicted = list(only_matched["predicted"])
# names = list(only_matched["name"])

# for name in names:
#     shutil.copy(os.path.join("madagascar_nonAnopheles_2", name), os.path.join("results/madagascar_nonAnopheles_2", "matched", name))


# for image_name in os.listdir("results/madagascar_nonAnopheles_2/matched"):

#     img = cv2.resize(cv2.imread("results/madagascar_nonAnopheles_2/matched/" + image_name), (224, 224))
#     x = image.img_to_array(img) / 255
#     res = model.predict(np.array([x]).astype('float32')).tolist()[0]
#     predicted = larvae[res.index(max(res))]
#     percentage = str(max(res) * 100)

#     img = cv2.imread("results/madagascar_nonAnopheles_2/matched/" + image_name)
#     height, width, _ = img.shape
#     threshold = 0
#     alpha = 0.5

#     ## cam
#     superimposed_img = all_cams.cam(model, grad_model, x, "efficientNet", height, width, threshold = threshold)
#     superimposed_img = superimposed_img * alpha + img
#     cv2.imwrite(f"results/madagascar_nonAnopheles_2/cam/{threshold}_cam_{image_name}", superimposed_img)

#     ## grad- cam
#     superimposed_img = all_cams.grad_cam_keras(model, x, height, width, threshold = threshold)
#     superimposed_img = superimposed_img * alpha + img
#     cv2.imwrite(f"results/madagascar_nonAnopheles_2/grad_cam/{threshold}_gradCam_{image_name}", superimposed_img)
    
#     ## grad- cam- mine
#     superimposed_img = all_cams.grad_cam_mine(grad_model, x, height, width, threshold = threshold)
#     superimposed_img = superimposed_img * alpha + img
#     cv2.imwrite(f"results/madagascar_nonAnopheles_2/grad_cam_mine/{threshold}_gradCam_{image_name}", superimposed_img)



# import pandas as pd
# import os
# import cv2

# import albumentations
# import albumentations.pytorch

# albumentations_transform = albumentations.Compose([
#     albumentations.Rotate(limit=(30, 60), border_mode = cv2.BORDER_CONSTANT, value = 0, always_apply=True),
#     albumentations.Rotate(limit=(-60, -30), border_mode = cv2.BORDER_CONSTANT, value = 0, always_apply=True),
#     albumentations.VerticalFlip(p=1, always_apply=True),
#     albumentations.HorizontalFlip(p=1, always_apply=True),
#     albumentations.Sharpen(alpha=(0.5, 1.0), lightness=(1.0, 1.0), p=1, always_apply=True),
#     albumentations.MedianBlur(blur_limit=[31, 51], p=1, always_apply=True),
#     albumentations.RandomBrightnessContrast(brightness_limit=[-0.2, 0.1], contrast_limit=[-0.2, 0.1], p=1, always_apply=True)
# ])

# aug_names = ["rotate_1", "rotate_2", "fliplr", "flipud", "sharp", "mediainBlur", "BrightnessContrast"]

# dataset = os.listdir("random_test_image_cdc/images")
# if ".DS_Store" in dataset: dataset.remove(".DS_Store")

# for i, name in enumerate(dataset): 
    
#     img_path= os.path.join("random_test_image_cdc/images", name)
#     img = cv2.imread(img_path)

#     aug_img = {}
#     for bla in range(7):
#         aug_img[aug_names[bla]] = albumentations_transform[bla](image = img)["image"]

#     for n in aug_names:
#         rename = f'{name.split(".jpg")[0]}_{n}.jpg'
#         cv2.imwrite(os.path.join("random_test_image_cdc/augmented_images", rename), aug_img[n])


