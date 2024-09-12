# Precise Localization of Mosquito Larvae Body Parts with Faster R-CNN Deep Learning
It presents an application of the Faster Region-based Convolutional Neural Network (Faster R-CNN) in the field of entomology and vector-borne disease control. we introduce an innovative approach for the automated and precise localization of the distinct body parts of mosquito larvae, namely the head, thorax, abdomen, and tail.

## Dataset Details

Our computer vision team at the University of South Florida gathered a dataset of 241 smartphone photos of verified specimens of An. stephensi and An. gambiae. The photos were taken via smartphone at the insectaries at USF and the Centers for Disease Control and Prevention (CDC).

![Dataset Link](https://drive.google.com/drive/folders/1Q6PDXFhdGPhoQmbvCyQmOp9JACOfPsjQ?usp=sharing)

![dataset.png](https://github.com/FarhatBuet14/mosquitoAI/blob/main/larvaeNET/Larvae%20Localization/larvae_anatomy_localization/images/dataset_details.png)

## Requirements
* Python 3.10.12
* Tensorflow 2.12
* Keras 2.12
* detectron2

### Model Architecture ans Training Methods

The primary objective was to develop a localization model using Faster R-CNN (Region-based Convolutional Neural Network) to precisely identify and delineate diagnostic areas of the mosquito larvae: head, thorax, abdomen, and lower region. The Faster R-CNN model architecture used in this work follows the typical structure consisting of a Region Proposal Network (RPN) and a Fast R-CNN detector. A pre-trained convolutional neural network ResNet was employed as the backbone for feature extraction, taking the input images and generating a feature map. The RPN, a fully convolutional network, operates on this feature map to generate region proposals (bounding boxes) along with class scores and bounding box regression coordinates for each proposal, using anchor boxes of varying scales and aspect ratios. These proposals are then filtered and refined using non-maximum suppression (NMS). The refined proposals are used to extract features from the feature map via RoI pooling, which applies max-pooling within each proposal region to obtain fixed spatial dimensions. These pooled features are fed into two fully connected layers for object classification and bounding box regression, predicting class probabilities and refining the bounding box coordinates, respectively. 

The model was trained end-to-end for 5,000 iterations using a subset of 164 photos, using a multi-task loss function combining classification, bounding box regression, and RPN losses, incorporating techniques like data augmentation (eight-fold; n=1,312), transfer learning, and appropriate optimization algorithms. During inference, the RPN generates proposals, which are processed by the Fast R-CNN detector to obtain the final object detections, classifications, and refined bounding boxes for the mosquito larvae body parts.

### Test Results

We have a test dataset comprising two specimens, totaling 42 images, fully separated during the training of the model. When tested on these test images, the effort resulted in *96.15%* accuracy in terms of average precision at IoU (Intersection-over-Union) of at least 50%.

![test_results.png](https://github.com/FarhatBuet14/mosquitoAI/blob/main/larvaeNET/Larvae%20Localization/larvae_anatomy_localization/images/test_results.png)

We are also hosting this localizaion test process in the [website](https://mosquito-localization.web.app/) where you can upload an image and check the results with the bounding boxes within less than a minute.